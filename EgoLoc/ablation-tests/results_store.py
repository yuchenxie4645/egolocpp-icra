"""Crash-resilient append-only JSONL result storage."""

import fcntl
import hashlib
import json
import os
from pathlib import Path

VALID_STATUSES = {"success", "terminal_failure"}


class ResultsStoreError(RuntimeError):
    """Base error for an invalid or incompatible results file."""


class DuplicateResultError(ResultsStoreError):
    """An immutable trial key was written more than once."""


class ConfigMismatchError(ResultsStoreError):
    """Existing records were produced with a different locked config."""


def canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def stable_config_hash(config):
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def result_key(episode, task, mode, trial):
    episode = f"{int(episode):06d}"
    return f"{episode}|{task}|{mode}|{int(trial)}"


def read_jsonl_tolerant(path):
    """Read JSONL while ignoring one non-newline-terminated final fragment."""
    path = Path(path)
    if not path.exists():
        return [], False
    payload = path.read_bytes()
    if not payload:
        return [], False
    lines = payload.splitlines(keepends=True)
    records = []
    torn = False
    for index, raw_line in enumerate(lines):
        final = index == len(lines) - 1
        terminated = raw_line.endswith((b"\n", b"\r"))
        if final and not terminated:
            torn = True
            break
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ResultsStoreError(
                f"malformed JSONL line {index + 1} in {path}"
            ) from exc
    return records, torn


class ResultsStore:
    """One fsync'd O_APPEND write per immutable trial record."""

    def __init__(self, path, config_hash=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config_hash = config_hash
        records, self._has_torn_final = read_jsonl_tolerant(self.path)
        self.records = records
        self.resume_index = {}
        for record in records:
            key = record.get("key")
            if not isinstance(key, str):
                raise ResultsStoreError("existing result is missing string key")
            if key in self.resume_index:
                raise DuplicateResultError(f"duplicate existing result key: {key}")
            status = record.get("status")
            if status not in VALID_STATUSES:
                raise ResultsStoreError(
                    f"existing result {key} has invalid status {status!r}"
                )
            found_hash = record.get("config_hash")
            if config_hash is not None and found_hash != config_hash:
                raise ConfigMismatchError(
                    f"existing result {key} config_hash={found_hash!r}; "
                    f"current={config_hash!r}. Use a new output path."
                )
            self.resume_index[key] = record

    @property
    def completed_keys(self):
        return set(self.resume_index)

    def __contains__(self, key):
        return key in self.resume_index

    def _remove_torn_tail_locked(self, fd):
        if not self._has_torn_final:
            return
        os.lseek(fd, 0, os.SEEK_SET)
        payload = b""
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            payload += chunk
        last_newline = payload.rfind(b"\n")
        truncate_to = last_newline + 1 if last_newline >= 0 else 0
        os.ftruncate(fd, truncate_to)
        os.fsync(fd)
        self._has_torn_final = False

    def append(self, record):
        record = dict(record)
        key = record.get("key")
        if not isinstance(key, str) or key.count("|") != 3:
            raise ResultsStoreError(f"invalid immutable result key: {key!r}")
        if key in self.resume_index:
            raise DuplicateResultError(f"result key already complete: {key}")
        if record.get("status") not in VALID_STATUSES:
            raise ResultsStoreError(
                f"status must be one of {sorted(VALID_STATUSES)}"
            )
        if self.config_hash is not None:
            supplied = record.get("config_hash")
            if supplied is None:
                record["config_hash"] = self.config_hash
            elif supplied != self.config_hash:
                raise ConfigMismatchError(
                    f"record config_hash={supplied!r}; "
                    f"store={self.config_hash!r}"
                )
        payload = (canonical_json(record) + "\n").encode("utf-8")
        fd = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_APPEND,
            0o664,
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            self._remove_torn_tail_locked(fd)
            view = memoryview(payload)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short append to result JSONL")
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self.resume_index[key] = record
        self.records.append(record)
        return record
