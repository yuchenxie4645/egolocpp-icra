#!/usr/bin/env bash
set -euo pipefail

SESSION="egoloc-occlubench"
READY="/home/EgoLoc/ablation-tests/data/.signals-ready"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Session ${SESSION} already exists; attaching."
  exec tmux attach-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" -n prepare \
  "docker exec xyc bash -c 'cd /home/EgoLoc/ablation-tests && rm -f ${READY} && conda run -n egolocxyc python run_full.py manifest && conda run -n egolocxyc python run_full.py convert && CUDA_VISIBLE_DEVICES=0 conda run -n egolocxyc python run_full.py signals --device cuda:0 && touch ${READY}'"

tmux new-window -t "${SESSION}" -n evaluate \
  "docker exec xyc bash -c 'cd /home/EgoLoc/ablation-tests && while [ ! -f ${READY} ]; do sleep 30; done && CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 conda run -n egolocxyc python run_full.py evaluate && conda run -n egolocxyc python run_full.py audit && conda run -n egolocxyc python run_full.py stats'"

tmux select-window -t "${SESSION}:prepare"
echo "Started ${SESSION}: prepare uses GPU 0; evaluation waits, then uses GPU 1."
exec tmux attach-session -t "${SESSION}"
