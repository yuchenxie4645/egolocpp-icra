#!/usr/bin/env bash
set -euo pipefail

if sudo -n true 2>/dev/null; then
  sudo apt-get update
  sudo apt-get install -y \
    git git-lfs wget curl unzip p7zip-full tar rsync tree ffmpeg \
    python3 python3-venv python3-pip \
    jq
else
  echo "WARNING: sudo is unavailable non-interactively; skipping apt-get install." >&2
  echo "Install missing packages manually if later commands report they are absent." >&2
fi

if command -v git-lfs >/dev/null 2>&1; then
  git lfs install
else
  echo "WARNING: git-lfs is not installed or not on PATH; skipping git lfs install." >&2
fi

mkdir -p scripts external reports
mkdir -p data/raw_downloads/usst data/raw_downloads/egoloc data/raw_downloads/egopat3d data/raw_downloads/desktil
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/odometry
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/trajectory_repair
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/video_clips_hand
mkdir -p data/EgoPAT3D/raw
mkdir -p data/DeskTIL/videos data/DeskTIL/annotations data/DeskTIL/raw
mkdir -p data/EgoLoc_samples/EgoPAT3D data/EgoLoc_samples/DeskTIL

if ! python3 -m venv .venv; then
  echo "WARNING: python3 -m venv failed; trying user-level virtualenv fallback." >&2
  python3 -m pip install --user -U virtualenv
  rm -rf .venv
  python3 -m virtualenv .venv
fi

source .venv/bin/activate

pip install -U pip wheel setuptools
pip install -U numpy pandas opencv-python pillow tqdm huggingface_hub datasets openpyxl
