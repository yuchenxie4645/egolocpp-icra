#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p external reports
mkdir -p data/raw_downloads/usst data/raw_downloads/egoloc data/raw_downloads/egopat3d data/raw_downloads/desktil
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/odometry
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/trajectory_repair
mkdir -p data/EgoPAT3D/EgoPAT3D-postproc/video_clips_hand
mkdir -p data/EgoPAT3D/raw data/EgoLoc_samples/EgoPAT3D data/EgoLoc_samples/DeskTIL
mkdir -p data/DeskTIL/videos data/DeskTIL/annotations data/DeskTIL/raw

BLOCKERS="reports/download_blockers.md"
touch "$BLOCKERS"

append_blocker() {
  local title="$1"
  local source="$2"
  local expected="$3"
  local problem="$4"
  local resolution="$5"

  {
    echo
    echo "## ${title}"
    echo
    echo "Source:"
    echo "${source}"
    echo
    echo "Expected:"
    echo "${expected}"
    echo
    echo "Problem:"
    echo "${problem}"
    echo
    echo "Resolution needed from human:"
    echo "${resolution}"
  } >> "$BLOCKERS"
}

clone_repo() {
  local url="$1"
  local dest="$2"
  if [ -d "$dest/.git" ]; then
    echo "Already cloned: $dest"
  else
    git clone "$url" "$dest"
  fi
}

clone_repo https://github.com/oppo-us-research/USST.git external/USST
clone_repo https://github.com/IRMVLab/EgoLoc.git external/EgoLoc
clone_repo https://github.com/ai4ce/EgoPAT3D.git external/EgoPAT3D_repo

{
  echo "# Cloned repository commits"
  echo
  for repo in external/USST external/EgoLoc external/EgoPAT3D_repo; do
    echo "## $repo"
    git -C "$repo" remote -v
    git -C "$repo" rev-parse HEAD
    echo
  done
} > reports/repo_commits.md

USST_ARCHIVE="data/raw_downloads/usst/EgoPAT3D-postproc.tar.gz"
if [ -f "$USST_ARCHIVE" ]; then
  tar -zxvf "$USST_ARCHIVE" -C data/EgoPAT3D/
else
  append_blocker \
    "USST EgoPAT3D-DT download blocker" \
    "https://github.com/oppo-us-research/USST" \
    "EgoPAT3D-postproc.tar.gz" \
    "The official USST data link must be obtained from the README and is hosted as a cloud file. Place a directly downloaded archive at ${USST_ARCHIVE}; this script will not use unofficial mirrors." \
    "Manually download EgoPAT3D-postproc.tar.gz from the USST README OneDrive link and place it at ${USST_ARCHIVE}. Then run: tar -zxvf ${USST_ARCHIVE} -C data/EgoPAT3D/"
fi

if [ "${FULL_EGOPAT3D_HF_DOWNLOAD:-0}" = "1" ]; then
  source .venv/bin/activate
  python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ai4ce/EgoPAT3Dv1",
    repo_type="dataset",
    local_dir="data/EgoPAT3D/raw/EgoPAT3Dv1",
    local_dir_use_symlinks=False,
)
PY
else
  append_blocker \
    "EgoPAT3D Hugging Face full snapshot not downloaded" \
    "https://huggingface.co/datasets/ai4ce/EgoPAT3Dv1" \
    "Official EgoPAT3D public dataset snapshot" \
    "FULL_EGOPAT3D_HF_DOWNLOAD was not set. The dataset may be large, so this script requires an explicit opt-in before downloading the full snapshot." \
    "After confirming disk capacity, run: FULL_EGOPAT3D_HF_DOWNLOAD=1 bash scripts/download_sources.sh"
fi

append_blocker \
  "EgoLoc EgoPAT3D sample download blocker" \
  "https://github.com/IRMVLab/EgoLoc" \
  "EgoPAT3D sample videos and manual annotations" \
  "EgoLoc sample packages are hosted on SJTU Pan links in the official README, which commonly require browser JavaScript or manual interaction. This script did not use unofficial mirrors." \
  "Manually download the EgoPAT3D sample package from the EgoLoc README into data/raw_downloads/egoloc/, then extract it into data/EgoLoc_samples/EgoPAT3D/."

append_blocker \
  "EgoLoc DeskTIL download blocker" \
  "https://github.com/IRMVLab/EgoLoc" \
  "DeskTIL sample videos and manual annotations" \
  "EgoLoc DeskTIL packages are hosted on SJTU Pan links in the official README, which commonly require browser JavaScript or manual interaction. This script did not use unofficial mirrors." \
  "Manually download the DeskTIL package from the EgoLoc README into data/raw_downloads/desktil/, extract it into data/EgoLoc_samples/DeskTIL/, then sort videos into data/DeskTIL/videos/ and annotations into data/DeskTIL/annotations/."
