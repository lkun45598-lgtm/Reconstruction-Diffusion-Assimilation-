#!/bin/bash
# usage: [KS=30,10,0,1] [WARMUP=0] [GPUS='5 6 7'] run_dps_test.sh OUTDIR  -> DPS (K=30,10,1) + free forecast on the 20 test cases x seeds 42,43,44, cycle from frame 0, GPUs 5/6/7, 20 processes
cd /data1/user/lz/RB_Diffusion_DA; D=$1; mkdir -p $D
G=(${GPUS:-5 6 7}); export NG=${#G[@]}; export GL="${G[*]}"
seq 0 19 | xargs -P 20 -I{} bash -c 'a=($GL); g=${a[$(({} % NG))]}; CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 KS='${KS:-30,10,0,1}' START=0 WARMUP='${WARMUP:-0}' ~/miniconda3/envs/dedalus3/bin/python analysis/dps_test.py cuda:0 '$D' 42:{} 43:{} 44:{} 2>&1 | grep -v UCX > '$D'/log_c{}.txt'
