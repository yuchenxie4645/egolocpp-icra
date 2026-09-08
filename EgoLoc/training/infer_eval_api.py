#!/usr/bin/env python3
"""Evaluate fixed external multimodal APIs on v2 OccluBench SFT grids.

The evaluator has exactly two code-defined OpenAI-compatible baselines:

* ``gpt-4o``: GPT-4o via ``https://api.chatanywhere.tech/v1``; key from
  lowercase environment variable ``openai``.
* ``iris``: Iris (``iris-thinking``) via ``https://arlowgpt.com/api``; key
  from lowercase environment variable ``constellations``.

Run inside the ``xyc`` container:

    conda activate egolocxyc
    python /home/EgoLoc/training/infer_eval_api.py --workers 4

Each model has an append-only JSONL checkpoint in ``api_eval_results`` and the
combined machine-readable report is ``summary.json``. Resume is the default:
every sample already present in a checkpoint, including a failed sample, is
skipped. Pass ``--retry-failed`` to append a new attempt for only the latest
failed samples, or ``--fresh`` to archive selected checkpoints before running.
Transient 429/5xx/timeout/connection failures are retried within one run; hard
4xx responses are checkpointed immediately and are not retried.
``--requests-per-second`` limits API attempt starts globally per model, including
retries; it does not limit completion rate, so slow requests may still overlap.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI

from grid_reward import gaussian_reward_for_cell, parse_pred_cell


TASKS = ("contact", "separation")
DEFAULT_ROOTS = {
    "contact": "/home/data_labeling/data/v2-occlubench-contact_dataset",
    "separation": "/home/data_labeling/data/v2-occlubench-separation_dataset",
}
REWARD_SIGMA = 1.0
SCHEMA_VERSION = 1
MAX_WORKERS = 32


@dataclass(frozen=True)
class ModelConfig:
    key: str
    label: str
    model_id: str
    base_url: str
    api_key_env: str
    checkpoint_name: str
    request_kwargs: Mapping[str, Any]


# Keep endpoint-specific request parameters here, not on the command line.
# In particular, this makes it straightforward to change max_tokens to an
# endpoint-specific token parameter without making model configuration mutable.
MODEL_CONFIGS = {
    "gpt-4o": ModelConfig(
        key="gpt-4o",
        label="GPT-4o",
        model_id="gpt-4o",
        base_url="https://api.chatanywhere.tech/v1",
        api_key_env="openai",
        checkpoint_name="gpt-4o.jsonl",
        request_kwargs={"max_tokens": 512},
    ),
    "iris": ModelConfig(
        key="iris",
        label="Iris",
        model_id="iris-thinking",
        base_url="https://arlowgpt.com/api",
        api_key_env="constellations",
        checkpoint_name="iris.jsonl",
        request_kwargs={"max_tokens": 8192},
    ),
}


class ConfigurationError(RuntimeError):
    """Raised before requests when local configuration is unusable."""


class RequestStartRateLimiter:
    """Globally space request-attempt starts without limiting completions."""

    def __init__(self, requests_per_second: float):
        if not math.isfinite(requests_per_second) or requests_per_second < 0:
            raise ValueError("requests_per_second must be finite and nonnegative")
        self.requests_per_second = requests_per_second
        self.minimum_interval_seconds = (
            1.0 / requests_per_second if requests_per_second else 0.0
        )
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait_for_start(self) -> None:
        """Wait for a request-start slot; zero means unlimited starts."""
        if self.minimum_interval_seconds == 0:
            return

        while True:
            with self._lock:
                now = time.monotonic()
                sleep_seconds = self._next_start - now
                if sleep_seconds <= 0:
                    self._next_start = now + self.minimum_interval_seconds
                    return
            # Never hold the scheduling lock while waiting.
            time.sleep(sleep_seconds)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def model_provenance(config: ModelConfig) -> dict[str, Any]:
    return {
        "key": config.key,
        "label": config.label,
        "id": config.model_id,
        "base_url": config.base_url,
        "request_kwargs": dict(config.request_kwargs),
    }


def make_sample_id(task: str, dataset_name: str, raw: dict[str, Any]) -> str:
    """Build a mount-independent ID from all request/scoring-relevant fields."""
    md = raw.get("metadata", raw)
    gt_cell = md.get("gt_cell", md.get("cell_number"))
    identity = {
        "task": task,
        "dataset_name": dataset_name,
        "file_name": raw.get("file_name"),
        "prompt": raw.get("prompt"),
        "video_source": md.get("video_source"),
        "video_index": md.get("video_index"),
        "gt_frame": md.get("gt_frame"),
        "gt_cell": gt_cell,
        "frame_indices": md.get("frame_indices"),
        "offset": md.get("offset"),
    }
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def load_validation_rows(task: str, dataset_root: str, limit: int) -> list[dict]:
    root = Path(dataset_root)
    metadata_path = root / "validation" / "metadata.jsonl"
    if not metadata_path.is_file():
        raise ConfigurationError(f"Validation metadata not found: {metadata_path}")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"Invalid JSON in {metadata_path}:{line_number}: {exc.msg}"
                ) from exc

            md = raw.get("metadata", raw)
            try:
                file_name = str(raw["file_name"])
                prompt = raw["prompt"]
                video_source = str(md["video_source"])
                video_index = int(md["video_index"])
                gt_frame = int(md["gt_frame"])
                frame_indices = [int(value) for value in md["frame_indices"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Invalid row in {metadata_path}:{line_number}: {exc}"
                ) from exc
            gt_cell_raw = md.get("gt_cell", md.get("cell_number"))
            if gt_cell_raw is None:
                raise ConfigurationError(
                    f"Missing gt_cell/cell_number in "
                    f"{metadata_path}:{line_number}"
                )
            try:
                gt_cell = int(gt_cell_raw)
            except (TypeError, ValueError) as exc:
                raise ConfigurationError(
                    f"Invalid gt_cell/cell_number in "
                    f"{metadata_path}:{line_number}: {gt_cell_raw!r}"
                ) from exc
            if not isinstance(prompt, str):
                raise ConfigurationError(
                    f"Prompt is not a string in {metadata_path}:{line_number}"
                )
            if not frame_indices:
                raise ConfigurationError(
                    f"Empty frame_indices in {metadata_path}:{line_number}"
                )
            if not 1 <= gt_cell <= len(frame_indices):
                raise ConfigurationError(
                    f"Ground-truth cell {gt_cell} is outside 1.."
                    f"{len(frame_indices)} in {metadata_path}:{line_number}"
                )

            sample_id = make_sample_id(task, root.name, raw)
            if sample_id in seen_ids:
                raise ConfigurationError(
                    f"Duplicate stable sample ID in {metadata_path}: "
                    f"{sample_id}"
                )
            seen_ids.add(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "task": task,
                    "dataset_root": str(root),
                    "dataset_name": root.name,
                    "metadata_path": str(metadata_path),
                    "metadata_line": line_number,
                    "row_index": len(rows),
                    "file_name": file_name,
                    "image_path": str(metadata_path.parent / file_name),
                    "prompt": prompt,
                    "video_source": video_source,
                    "video_index": video_index,
                    "offset": md.get("offset"),
                    "gt_frame": gt_frame,
                    "gt_cell": gt_cell,
                    "frame_indices": frame_indices,
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def to_jsonable(value: Any) -> Any:
    """Convert SDK/Pydantic values to lossless-enough JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return to_jsonable(model_dump(mode="json"))
        except TypeError:
            return to_jsonable(model_dump())
    if isinstance(value, Path):
        return str(value)
    return str(value)


def normalize_content(value: Any) -> str:
    """Normalize string or text-block final content without adding reasoning."""
    value = to_jsonable(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "".join(normalize_content(item) for item in value)
    if isinstance(value, dict):
        if "text" in value:
            text = value["text"]
            if isinstance(text, dict) and "value" in text:
                text = text["value"]
            return normalize_content(text)
        if "content" in value:
            return normalize_content(value["content"])
        if "value" in value:
            return normalize_content(value["value"])
        return ""
    return str(value)


def extract_reasoning_content(message: Any) -> tuple[Any, str | None]:
    """Return separately exposed reasoning_content, including model extras."""
    candidates: list[Any] = []
    direct = getattr(message, "reasoning_content", None)
    if direct is not None:
        candidates.append(direct)

    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, Mapping) and "reasoning_content" in model_extra:
        candidates.append(model_extra["reasoning_content"])

    dumped = to_jsonable(message)
    if isinstance(dumped, dict) and "reasoning_content" in dumped:
        candidates.append(dumped["reasoning_content"])

    for candidate in candidates:
        if candidate is not None:
            raw = to_jsonable(candidate)
            return raw, normalize_content(raw)
    return None, None


def redact(text: str, secret: str | None) -> str:
    if secret:
        text = text.replace(secret, "[REDACTED]")
    return text[:4000]


def exception_details(exc: Exception, secret: str | None) -> dict[str, Any]:
    status_code = getattr(exc, "status_code", None)
    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None
    request_id = getattr(exc, "request_id", None)
    return {
        "type": type(exc).__name__,
        "status_code": status_code,
        "request_id": str(request_id) if request_id is not None else None,
        "message": redact(str(exc), secret),
    }


def is_transient_error(exc: Exception, status_code: int | None) -> bool:
    if status_code == 429 or status_code == 408:
        return True
    if status_code is not None:
        return status_code >= 500
    name = type(exc).__name__.lower()
    return (
        isinstance(exc, TimeoutError)
        or "timeout" in name
        or "connection" in name
        or "connecterror" in name
    )


def score_content(row: dict, final_content: str) -> tuple[dict, dict]:
    """Score only normalized message.content, never reasoning_content."""
    pred_cell = parse_pred_cell(final_content)
    num_cells = len(row["frame_indices"])
    valid = pred_cell is not None and 1 <= pred_cell <= num_cells
    pred_frame = (
        int(row["frame_indices"][pred_cell - 1]) if valid else None
    )
    abs_frame_error = (
        abs(pred_frame - row["gt_frame"]) if pred_frame is not None else None
    )
    prediction = {
        "cell": pred_cell,
        "frame": pred_frame,
    }
    metrics = {
        "parse_success": pred_cell is not None,
        "valid_cell": valid,
        "exact_cell": pred_cell == row["gt_cell"],
        "absolute_frame_error": abs_frame_error,
        "gaussian_reward_sigma1": gaussian_reward_for_cell(
            pred_cell,
            row["gt_cell"],
            num_cells=num_cells,
            sigma=REWARD_SIGMA,
        ),
    }
    return prediction, metrics


def base_record(config: ModelConfig, row: dict) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "status": None,
        "completed_at": None,
        "model": model_provenance(config),
        "task": row["task"],
        "video_source": row["video_source"],
        "dataset": {
            "name": row["dataset_name"],
            "root": row["dataset_root"],
            "metadata_path": row["metadata_path"],
            "metadata_line": row["metadata_line"],
        },
        "sample": {
            "row_index": row["row_index"],
            "file_name": row["file_name"],
            "image_path": row["image_path"],
            "video_index": row["video_index"],
            "offset": row["offset"],
            "image_bytes": None,
            "image_sha256": None,
        },
        "prompt": row["prompt"],
        "ground_truth": {
            "cell": row["gt_cell"],
            "frame": row["gt_frame"],
            "frame_indices": row["frame_indices"],
        },
        "prediction": None,
        "metrics": None,
        "response": None,
        "latency_seconds": None,
        "attempt_count": 0,
        "attempts": [],
        "error": None,
    }


def finish_failure(
    record: dict[str, Any],
    row: dict,
    error: dict[str, Any],
    attempts: list[dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    prediction, metrics = score_content(row, "")
    record.update(
        {
            "status": "error",
            "completed_at": utc_now(),
            "prediction": prediction,
            "metrics": metrics,
            "latency_seconds": round(time.perf_counter() - started, 6),
            "attempt_count": len(attempts),
            "attempts": attempts,
            "error": error,
        }
    )
    return record


def evaluate_row(
    client: OpenAI,
    config: ModelConfig,
    row: dict,
    rate_limiter: RequestStartRateLimiter,
    secret: str,
    timeout_seconds: float,
    max_attempts: int,
    backoff_base: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    record = base_record(config, row)
    image_path = Path(row["image_path"])

    if not image_path.is_file():
        return finish_failure(
            record,
            row,
            {
                "type": "ImageNotFound",
                "status_code": None,
                "request_id": None,
                "message": f"Image not found: {image_path}",
            },
            [],
            started,
        )

    try:
        image_bytes = image_path.read_bytes()
    except OSError as exc:
        return finish_failure(
            record,
            row,
            exception_details(exc, secret),
            [],
            started,
        )
    record["sample"]["image_bytes"] = len(image_bytes)
    record["sample"]["image_sha256"] = hashlib.sha256(image_bytes).hexdigest()
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return finish_failure(
            record,
            row,
            {
                "type": "InvalidPNG",
                "status_code": None,
                "request_id": None,
                "message": f"Image does not have a PNG signature: {image_path}",
            },
            [],
            started,
        )

    image_url = (
        "data:image/png;base64,"
        + base64.b64encode(image_bytes).decode("ascii")
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
                {"type": "text", "text": row["prompt"]},
            ],
        }
    ]
    attempts: list[dict[str, Any]] = []

    for attempt_number in range(1, max_attempts + 1):
        rate_limiter.wait_for_start()
        attempt_started = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=config.model_id,
                messages=messages,
                timeout=timeout_seconds,
                **dict(config.request_kwargs),
            )
            attempt_latency = time.perf_counter() - attempt_started
            if not response.choices:
                raise ValueError("Chat Completions response has no choices")
            choice = response.choices[0]
            message = choice.message
            raw_content = to_jsonable(message.content)
            final_content = normalize_content(message.content)
            reasoning_raw, reasoning_content = extract_reasoning_content(message)
            prediction, metrics = score_content(row, final_content)
            attempts.append(
                {
                    "attempt": attempt_number,
                    "status": "success",
                    "latency_seconds": round(attempt_latency, 6),
                }
            )
            record.update(
                {
                    "status": "success",
                    "completed_at": utc_now(),
                    "prediction": prediction,
                    "metrics": metrics,
                    "response": {
                        "id": getattr(response, "id", None),
                        "content": final_content,
                        "content_raw": raw_content,
                        "reasoning_content": reasoning_content,
                        "reasoning_content_raw": reasoning_raw,
                        "finish_reason": getattr(
                            choice, "finish_reason", None
                        ),
                        "usage": to_jsonable(
                            getattr(response, "usage", None)
                        ),
                    },
                    "latency_seconds": round(
                        time.perf_counter() - started, 6
                    ),
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                    "error": None,
                }
            )
            return record
        except Exception as exc:
            attempt_latency = time.perf_counter() - attempt_started
            details = exception_details(exc, secret)
            transient = is_transient_error(exc, details["status_code"])
            will_retry = transient and attempt_number < max_attempts
            attempt_record = {
                "attempt": attempt_number,
                "status": "error",
                "latency_seconds": round(attempt_latency, 6),
                "retryable": transient,
                "will_retry": will_retry,
                "error": details,
            }
            if will_retry:
                delay = backoff_base * (2 ** (attempt_number - 1))
                delay += random.uniform(0.0, min(1.0, delay * 0.25))
                attempt_record["backoff_seconds"] = round(delay, 6)
                attempts.append(attempt_record)
                time.sleep(delay)
                continue
            attempts.append(attempt_record)
            return finish_failure(record, row, details, attempts, started)

    # The loop always returns, but keep a defensive checkpointable fallback.
    return finish_failure(
        record,
        row,
        {
            "type": "RetryLoopExhausted",
            "status_code": None,
            "request_id": None,
            "message": "Request retry loop ended unexpectedly",
        },
        attempts,
        started,
    )


class CheckpointWriter:
    """Append JSONL records durably under a write lock."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb+") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.seek(0, os.SEEK_END)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        self._handle = self.path.open("a", encoding="utf-8")

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(
            to_jsonable(record),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._lock:
            self._handle.write(line + "\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            self._handle.close()

    def __enter__(self) -> "CheckpointWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def load_latest_records(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[checkpoint] warning: ignoring malformed "
                    f"{path.name}:{line_number}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            sample_id = record.get("sample_id")
            if isinstance(sample_id, str):
                latest[sample_id] = record
    return latest


def archive_for_fresh(
    output_dir: Path, selected_configs: list[ModelConfig]
) -> Path | None:
    candidates = [
        output_dir / config.checkpoint_name for config in selected_configs
    ]
    candidates.append(output_dir / "summary.json")
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive_dir = output_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def run_model(
    config: ModelConfig,
    rows: list[dict],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_path = Path(args.output_dir) / config.checkpoint_name
    rate_limiter = RequestStartRateLimiter(args.requests_per_second)
    latest = load_latest_records(checkpoint_path)
    pending = []
    skipped_successes = 0
    skipped_failures = 0
    for row in rows:
        prior = latest.get(row["sample_id"])
        if prior is None:
            pending.append(row)
        elif args.retry_failed and prior.get("status") != "success":
            pending.append(row)
        elif prior.get("status") == "success":
            skipped_successes += 1
        else:
            skipped_failures += 1

    print(
        f"\n[model] {config.label} | id={config.model_id} | "
        f"endpoint={config.base_url}",
        flush=True,
    )
    if rate_limiter.requests_per_second:
        print(
            f"[rate] request-start limit="
            f"{rate_limiter.requests_per_second:g}/s globally across workers "
            f"(minimum spacing {rate_limiter.minimum_interval_seconds:g}s; "
            f"retries included)",
            flush=True,
        )
    else:
        print(
            "[rate] request-start limit=unlimited (retries included)",
            flush=True,
        )
    print(
        f"[resume] selected={len(rows)} pending={len(pending)} "
        f"skipped_success={skipped_successes} "
        f"skipped_failure={skipped_failures}",
        flush=True,
    )

    if pending:
        secret = os.environ.get(config.api_key_env)
        if not secret:
            raise ConfigurationError(
                f"Required environment variable {config.api_key_env!r} is "
                f"not set for {config.label}; no requests were sent."
            )
        client = OpenAI(
            api_key=secret,
            base_url=config.base_url,
            timeout=args.timeout,
            max_retries=0,
        )
        completed = 0
        try:
            with CheckpointWriter(checkpoint_path) as writer:
                with ThreadPoolExecutor(
                    max_workers=args.workers,
                    thread_name_prefix=f"eval-{config.key}",
                ) as executor:
                    future_rows = {
                        executor.submit(
                            evaluate_row,
                            client,
                            config,
                            row,
                            rate_limiter,
                            secret,
                            args.timeout,
                            args.max_attempts,
                            args.backoff_base,
                        ): row
                        for row in pending
                    }
                    for future in as_completed(future_rows):
                        row = future_rows[future]
                        try:
                            record = future.result()
                        except Exception as exc:
                            started = time.perf_counter()
                            record = finish_failure(
                                base_record(config, row),
                                row,
                                exception_details(exc, secret),
                                [],
                                started,
                            )
                        writer.append(record)
                        latest[row["sample_id"]] = record
                        completed += 1
                        if record["status"] == "error":
                            error = record.get("error") or {}
                            status = error.get("status_code")
                            print(
                                f"[{config.key}] {completed}/{len(pending)} "
                                f"ERROR sample={row['sample_id'][:12]} "
                                f"type={error.get('type')} "
                                f"http={status if status is not None else '-'}",
                                file=sys.stderr,
                                flush=True,
                            )
                        elif (
                            completed == 1
                            or completed == len(pending)
                            or completed % 25 == 0
                        ):
                            print(
                                f"[{config.key}] completed "
                                f"{completed}/{len(pending)}",
                                flush=True,
                            )
        finally:
            client.close()

    selected_records: list[dict[str, Any]] = []
    missing_ids: list[str] = []
    for row in rows:
        record = latest.get(row["sample_id"])
        if record is None:
            missing_ids.append(row["sample_id"])
        else:
            selected_records.append(record)
    if missing_ids:
        raise ConfigurationError(
            f"{len(missing_ids)} selected rows are missing from "
            f"{checkpoint_path}"
        )
    resume = {
        "checkpoint_path": str(checkpoint_path),
        "selected_rows": len(rows),
        "requested_this_run": len(pending),
        "skipped_successes": skipped_successes,
        "skipped_failures": skipped_failures,
        "retry_failed": bool(args.retry_failed),
    }
    return selected_records, resume


def new_counter() -> dict[str, Any]:
    return {
        "n": 0,
        "succeeded": 0,
        "failed": 0,
        "parsed": 0,
        "valid": 0,
        "correct": 0,
        "absolute_frame_error": 0.0,
        "frame_mae_n": 0,
        "gaussian_reward": 0.0,
    }


def add_record(counter: dict[str, Any], record: dict[str, Any]) -> None:
    counter["n"] += 1
    if record.get("status") == "success":
        counter["succeeded"] += 1
    else:
        counter["failed"] += 1
    metrics = record.get("metrics") or {}
    counter["parsed"] += int(bool(metrics.get("parse_success")))
    counter["valid"] += int(bool(metrics.get("valid_cell")))
    counter["correct"] += int(bool(metrics.get("exact_cell")))
    frame_error = metrics.get("absolute_frame_error")
    if frame_error is not None:
        counter["absolute_frame_error"] += float(frame_error)
        counter["frame_mae_n"] += 1
    counter["gaussian_reward"] += float(
        metrics.get("gaussian_reward_sigma1") or 0.0
    )


def finalized_metrics(counter: dict[str, Any]) -> dict[str, Any]:
    n = counter["n"]
    mae_n = counter["frame_mae_n"]
    return {
        "count": n,
        "successful_requests": counter["succeeded"],
        "failed_requests": counter["failed"],
        "parse_success_count": counter["parsed"],
        "parse_success_rate": counter["parsed"] / n if n else None,
        "valid_cell_count": counter["valid"],
        "valid_cell_rate": counter["valid"] / n if n else None,
        "exact_cell_count": counter["correct"],
        "exact_cell_accuracy": counter["correct"] / n if n else None,
        "frame_mae": (
            counter["absolute_frame_error"] / mae_n if mae_n else None
        ),
        "frame_mae_count": mae_n,
        "sigma1_gaussian_reward_sum": counter["gaussian_reward"],
        "sigma1_gaussian_reward_mean": (
            counter["gaussian_reward"] / n if n else None
        ),
    }


def summarize_model(
    config: ModelConfig,
    records: list[dict[str, Any]],
    resume: dict[str, Any],
) -> dict[str, Any]:
    by_task_source: defaultdict[tuple[str, str], dict[str, Any]]
    by_task_source = defaultdict(new_counter)
    by_task: defaultdict[str, dict[str, Any]] = defaultdict(new_counter)
    overall = new_counter()
    for record in records:
        task = str(record.get("task", "?"))
        source = str(record.get("video_source", "?"))
        add_record(by_task_source[(task, source)], record)
        add_record(by_task[task], record)
        add_record(overall, record)

    return {
        "model": model_provenance(config),
        "resume": resume,
        "overall": finalized_metrics(overall),
        "by_task_source": [
            {
                "task": task,
                "source": source,
                **finalized_metrics(counter),
            }
            for (task, source), counter in sorted(by_task_source.items())
        ],
        "by_task": [
            {"task": task, **finalized_metrics(counter)}
            for task, counter in sorted(by_task.items())
        ],
    }


def print_model_report(summary: dict[str, Any]) -> None:
    model = summary["model"]
    print("\n" + "=" * 92)
    print(
        f"{model['label']} | model={model['id']} | endpoint={model['base_url']}"
    )
    print(
        f"{'task/source':<30}{'n':>6}{'fail':>7}{'parse%':>10}"
        f"{'cell_acc%':>12}{'frameMAE':>12}{'avgReward':>12}"
    )
    print("-" * 92)

    def print_line(label: str, metrics: dict[str, Any]) -> None:
        parse_rate = metrics["parse_success_rate"]
        accuracy = metrics["exact_cell_accuracy"]
        frame_mae = metrics["frame_mae"]
        reward = metrics["sigma1_gaussian_reward_mean"]
        parse_text = (
            f"{100.0 * parse_rate:.1f}%" if parse_rate is not None else "n/a"
        )
        accuracy_text = (
            f"{100.0 * accuracy:.1f}%" if accuracy is not None else "n/a"
        )
        mae_text = f"{frame_mae:.3f}" if frame_mae is not None else "nan"
        reward_text = f"{reward:.4f}" if reward is not None else "nan"
        print(
            f"{label:<30}{metrics['count']:>6}"
            f"{metrics['failed_requests']:>7}{parse_text:>10}"
            f"{accuracy_text:>12}{mae_text:>12}{reward_text:>12}"
        )

    for item in summary["by_task_source"]:
        print_line(f"{item['task']}/{item['source']}", item)
    print("-" * 92)
    for item in summary["by_task"]:
        print_line(f"{item['task']} (all)", item)
    print("=" * 92, flush=True)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            to_jsonable(value),
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def bounded_workers(value: str) -> int:
    workers = int(value)
    if not 1 <= workers <= MAX_WORKERS:
        raise argparse.ArgumentTypeError(
            f"workers must be between 1 and {MAX_WORKERS}"
        )
    return workers


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(
            "value must be finite and nonnegative"
        )
    return number


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS),
        default=list(MODEL_CONFIGS),
        help=(
            "Fixed baseline(s) to evaluate (default: gpt-4o iris). "
            "Model IDs, endpoints, and key variable names are code-defined."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=TASKS,
        default=list(TASKS),
        help="Task dataset(s) to evaluate (default: both).",
    )
    parser.add_argument(
        "--contact-root",
        default=DEFAULT_ROOTS["contact"],
        help="Contact dataset root containing validation/metadata.jsonl.",
    )
    parser.add_argument(
        "--separation-root",
        default=DEFAULT_ROOTS["separation"],
        help="Separation dataset root containing validation/metadata.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "api_eval_results"),
        help="Checkpoint/report directory (default: %(default)s).",
    )
    parser.add_argument(
        "--limit",
        type=nonnegative_int,
        default=0,
        help="If >0, use the first N metadata rows per task.",
    )
    parser.add_argument(
        "--workers",
        type=bounded_workers,
        default=4,
        help=f"Concurrent requests per model, 1..{MAX_WORKERS} (default: 4).",
    )
    parser.add_argument(
        "--requests-per-second",
        type=nonnegative_float,
        default=0.0,
        help=(
            "Maximum API request-attempt starts per second, globally across "
            "workers for each model run. Retries count; completion rate is "
            "not limited. Use 0 for unlimited (default: 0)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=positive_float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--max-attempts",
        type=positive_int,
        default=4,
        help=(
            "Maximum attempts for transient 429/5xx/timeout/connection "
            "failures (default: 4). Hard 4xx responses get one attempt."
        ),
    )
    parser.add_argument(
        "--backoff-base",
        type=positive_float,
        default=2.0,
        help="Initial exponential retry delay in seconds (default: 2).",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "On resume, append a new attempt for rows whose latest "
            "checkpoint record failed. By default failures count as "
            "completed and are not called again."
        ),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Archive selected model checkpoints and summary.json under "
            "<output-dir>/archive/<timestamp>/ before evaluating."
        ),
    )
    return parser.parse_args()


def deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main() -> int:
    args = parse_args()
    args.models = deduplicate(args.models)
    args.tasks = deduplicate(args.tasks)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_configs = [MODEL_CONFIGS[key] for key in args.models]

    if args.fresh:
        archive_dir = archive_for_fresh(output_dir, selected_configs)
        if archive_dir is not None:
            print(f"[fresh] archived prior results to {archive_dir}", flush=True)
        else:
            print("[fresh] no prior selected results to archive", flush=True)

    roots = {
        "contact": args.contact_root,
        "separation": args.separation_root,
    }
    rows_by_task = {
        task: load_validation_rows(task, roots[task], args.limit)
        for task in args.tasks
    }
    rows = [
        row
        for task in args.tasks
        for row in rows_by_task[task]
    ]
    print(
        "[data] "
        + " ".join(
            f"{task}={len(rows_by_task[task])}" for task in args.tasks
        )
        + f" total={len(rows)} limit_per_task={args.limit or 'all'}",
        flush=True,
    )

    model_summaries = []
    for config in selected_configs:
        records, resume = run_model(config, rows, args)
        model_summary = summarize_model(config, records, resume)
        model_summaries.append(model_summary)
        print_model_report(model_summary)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "evaluator": str(Path(__file__).resolve()),
        "reward_sigma": REWARD_SIGMA,
        "settings": {
            "models": args.models,
            "tasks": args.tasks,
            "limit_per_task": args.limit,
            "workers_per_model": args.workers,
            "requests_per_second": args.requests_per_second,
            "timeout_seconds": args.timeout,
            "max_attempts": args.max_attempts,
            "backoff_base_seconds": args.backoff_base,
            "retry_failed": bool(args.retry_failed),
            "fresh": bool(args.fresh),
        },
        "datasets": {
            task: {
                "root": roots[task],
                "selected_rows": len(rows_by_task[task]),
            }
            for task in args.tasks
        },
        "models": model_summaries,
    }
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"\n[summary] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
