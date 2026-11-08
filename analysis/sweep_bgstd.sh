#!/bin/bash
# usage: [NPROC=20] [SEED=42] [KS=30,10] [GPUS='2 3 6 7'] sweep_bgstd.sh OUTROOT "BGSTD1 BGSTD2 ..." "CASES" -> validation sweep of the DPS background std, other-case-init setting, K=10 and 30, seed 42, one process per (bg_std, case)
cd /data1/user/lz/RB_Diffusion_DA; R=$1; n=0; GL=(${GPUS:-0 1 2 3 4 5 6 7})
for B in $2; do for c in $3; do g=${GL[$((n % ${#GL[@]}))]}; n=$((n+1)); D=$R/bg$B; mkdir -p $D
  echo "for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 INIT=other SPLIT=val BG_STD=$B KS=${KS:-30,10} START=0 ~/miniconda3/envs/dedalus3/bin/python analysis/dps_test.py cuda:0 $D ${SEED:-42}:$c 2>&1 | grep -v UCX >> $D/log_c$c.txt; done"; done; done | xargs -P ${NPROC:-20} -I{} bash -c "{}"
