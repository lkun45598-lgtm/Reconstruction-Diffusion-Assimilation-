#!/bin/bash
# usage: rbc_dps_start.sh SESSION JOBS NSLOTS CONC PYTHON   在 tmux 里启动 rbc_dps_queue.py（每槽 5 核：0-4,5-9,...）
S=$1; J=$2; N=$3; C=$4; PY=$5; cd "$(dirname "$0")/.."; SL=""; for i in $(seq 0 $((N-1))); do SL="$SL$((i*5))-$((i*5+4)),"; done
tmux kill-session -t $S 2>/dev/null; tmux new-session -d -s $S "cd $(pwd); $PY analysis/rbc_dps_queue.py --jobs $J --slots ${SL%,} --min-free 11000 --conc $C --python $PY 2>&1 | tee -a queue_tune.log"; sleep 2; tmux ls | cut -d: -f1 | tr '\n' ' '
