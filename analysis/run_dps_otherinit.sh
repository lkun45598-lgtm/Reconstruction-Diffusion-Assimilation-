#!/bin/bash
# usage: [NPROC=8] [GPUS='2 3 6 7'] run_dps_otherinit.sh OUTDIR  -> test set, new setting (cycle-0 background = last of the 20 solver frames from the EnKF donor):
# DPS K=10 (bg_std 0.5), DPS K=30 (bg_std 1), free forecast (K=0), 20 cases x seeds 42,43,44. dps_test.py skips finished csv, so each job is retried up to 3 times (GPU memory is shared).
cd /data1/user/lz/RB_Diffusion_DA; D=$1; mkdir -p $D; GL=(${GPUS:-2 3 6 7}); n=0
for spec in "10 0.5" "30 1" "0 0.5"; do set -- $spec; for c in $(seq 0 19); do g=${GL[$((n % ${#GL[@]}))]}; n=$((n+1))
 echo "for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 INIT=other SPLIT=test BG_STD=$2 KS=$1 START=0 ~/miniconda3/envs/dedalus3/bin/python analysis/dps_test.py cuda:0 $D 42:$c 43:$c 44:$c 2>&1 | grep -v UCX >> $D/log_K$1_c$c.txt; done"; done; done | xargs -P ${NPROC:-8} -I{} bash -c "{}"
