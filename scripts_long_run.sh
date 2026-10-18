#!/bin/bash
# long-horizon check: seeds 81-83 (= test cases 0-2) regenerated to 520 frames, frames 120..519 = 400 cycles
cd /data1/user/lz/RB_Diffusion_DA
PY=/home/lz/miniconda3/envs/dedalus3/bin/python
for s in 81 82 83; do while [ ! -f data/long_raw/seed_$s/case_00000.npy ]; do sleep 30; done; done
mkdir -p data/processed_long/test outputs/study_rb_long
cp data/processed/stats.npz data/processed_long/
$PY - <<'P'
import numpy as np
for i,s in enumerate((81,82,83)):
    a=np.load(f'data/long_raw/seed_{s}/case_00000.npy'); old=np.load(f'data/processed/test/case_0000{i}.npy')
    print(s,a.shape,'max |new-old| on the first 100 frames:',float(np.abs(a[120:220]-old).max()),flush=True)
    np.save(f'data/processed_long/test/case_0000{i}.npy',a[120:].astype(np.float32))
P
DP="--init noise --bg-std 0.3 --dps-rate 2 --obs-std 0.05 --no-hard-observation"
for ci in 0 1 2; do
  g=$(( ci==0 ? 0 : ci==1 ? 2 : 5 )); d=outputs/study_rb_long/c$ci; mkdir -p $d
  tmux new-session -d -s long$ci "OMP_NUM_THREADS=1 $PY analysis/study_rb.py enkf --data data/processed_long --split test --case-index $ci --out $d --analysis-steps 399 --enkf-members 20 --enkf-obs-std 0.01 --inflation 2.5 --radius-x 1.5 --radius-z 1.2 --every 3 --schemes dps enkf_infl_loc $DP --device cuda:$g 2>&1 | grep -v 'UCX\|DEBUG\|INFO' > $d/run.log"
done
