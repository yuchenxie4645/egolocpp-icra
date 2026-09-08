#!/usr/bin/env python3
"""Resumable stage driver for the OccluBench-only ablation."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "data" / "manifest.json"
DEFAULT_RESULTS = HERE / "results" / "trials.jsonl"


def _run(command, stage, env=None, logs_dir=None):
    logs_dir = Path(logs_dir or HERE / "logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = logs_dir / f"{timestamp}_{stage}.log"
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(json.dumps({"stage": stage, "command": command}) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=HERE,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
        log.write(json.dumps({"exit_code": return_code}) + "\n")
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return str(log_path)


def _episode_args(episodes):
    return ["--episodes", *episodes] if episodes else []


def run(args):
    python = sys.executable
    logs = []
    partial = bool(args.episodes or args.limit_episodes)

    commands = {
        "manifest": [
            [
                python,
                str(HERE / "manifest.py"),
                "build",
                "--output",
                args.manifest,
            ],
            [
                python,
                str(HERE / "manifest.py"),
                "validate",
                "--output",
                args.manifest,
            ],
        ],
        "convert": [
            [
                python,
                str(HERE / "convert.py"),
                "--manifest",
                args.manifest,
                "--workers",
                str(args.workers),
                *(
                    ["--limit", str(args.limit_episodes)]
                    if args.limit_episodes is not None
                    else []
                ),
                *_episode_args(args.episodes),
                *(["--force"] if args.force else []),
            ]
        ],
        "signals": [
            [
                python,
                str(HERE / "signals.py"),
                "--manifest",
                args.manifest,
                "--device",
                args.device,
                *(
                    ["--limit", str(args.limit_episodes)]
                    if args.limit_episodes is not None
                    else []
                ),
                *_episode_args(args.episodes),
                *(["--force"] if args.force else []),
                *(["--debug"] if args.debug else []),
            ]
        ],
        "evaluate": [
            [
                python,
                str(HERE / "runner.py"),
                "--manifest",
                args.manifest,
                "--results",
                args.results,
                *(
                    ["--limit-episodes", str(args.limit_episodes)]
                    if args.limit_episodes is not None
                    else []
                ),
                *_episode_args(args.episodes),
                *(["--dry-run"] if args.dry_run else []),
            ]
        ],
        "audit": [
            [
                python,
                str(HERE / "audit.py"),
                "--manifest",
                args.manifest,
                "--results",
                args.results,
                *(["--partial"] if partial else []),
            ]
        ],
        "stats": [
            [
                python,
                str(HERE / "stats.py"),
                "--manifest",
                args.manifest,
                "--results",
                args.results,
                "--bootstrap-samples",
                str(args.bootstrap_samples),
                *(["--allow-partial"] if partial else []),
            ]
        ],
    }
    order = (
        ["manifest", "convert", "signals", "evaluate", "audit", "stats"]
        if args.stage == "all"
        else [args.stage]
    )
    if args.dry_run and any(stage in ("audit", "stats") for stage in order):
        raise ValueError(
            "--dry-run writes a separate fake-backend smoke file; invoke only "
            "the evaluate stage for dry-run"
        )
    for stage in order:
        stage_env = None
        if stage == "signals":
            stage_env = {"CUDA_VISIBLE_DEVICES": "0"}
        elif stage == "evaluate":
            stage_env = {
                "CUDA_VISIBLE_DEVICES": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        for index, command in enumerate(commands[stage]):
            label = stage if len(commands[stage]) == 1 else f"{stage}_{index + 1}"
            logs.append(
                _run(
                    command,
                    label,
                    env=stage_env,
                    logs_dir=args.logs_dir,
                )
            )
    return logs


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "manifest",
            "convert",
            "signals",
            "evaluate",
            "audit",
            "stats",
            "all",
        ],
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS))
    parser.add_argument("--episodes", nargs="*")
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--logs-dir", default=str(HERE / "logs"))
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    logs = run(args)
    print(json.dumps({"stage": args.stage, "logs": logs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
