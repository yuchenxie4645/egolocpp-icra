#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(".").resolve()
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)

SKIP_PARTS = {".git", ".venv", "__pycache__"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_info(repo: Path):
    if not (repo / ".git").exists():
        return None
    try:
        remote = subprocess.check_output(["git", "-C", str(repo), "remote", "-v"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        return {"path": str(repo.relative_to(ROOT)), "commit": commit, "remote": remote}
    except Exception as e:
        return {"path": str(repo), "error": repr(e)}


def main():
    files = []
    total_bytes = 0

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue

        rel = path.relative_to(ROOT)
        size = path.stat().st_size
        total_bytes += size

        item = {
            "path": str(rel),
            "size_bytes": size,
            "suffix": path.suffix.lower(),
        }

        # Hash small/medium metadata files. Do not hash huge videos unless needed.
        if size <= 100 * 1024 * 1024:
            item["sha256"] = sha256_file(path)
        else:
            item["sha256"] = None
            item["sha256_note"] = "Skipped because file is larger than 100MB."

        files.append(item)

    files.sort(key=lambda x: x["path"])

    repos = []
    for repo in [ROOT / "external" / "USST", ROOT / "external" / "EgoLoc", ROOT / "external" / "EgoPAT3D_repo"]:
        info = git_info(repo)
        if info:
            repos.append(info)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "total_files": len(files),
        "total_bytes": total_bytes,
        "repositories": repos,
        "files": files,
    }

    (REPORTS / "phase1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {REPORTS / 'phase1_manifest.json'}")
    print(f"Total files: {len(files)}")
    print(f"Total bytes: {total_bytes}")


if __name__ == "__main__":
    main()
