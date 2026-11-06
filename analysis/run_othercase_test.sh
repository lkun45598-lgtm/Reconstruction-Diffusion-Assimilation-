#!/bin/bash
# usage: run_othercase_test.sh K INFL SEED  -> EnKF (other-case init) on the 20 test cases, 600 frames, 20 in parallel
cd /data1/user/lz/RB_Diffusion_DA; K=$1; I=$2; S=$3; D=outputs/study_rb_othercase_test; mkdir -p $D
seq 0 19 | xargs -P 20 -I{} bash -c "OMP_NUM_THREADS=1 STEPS=599 KS=$K INFL=$I ~/miniconda3/envs/dedalus3/bin/python analysis/other_case_init.py test $S $D {} 2>&1 | grep -v UCX > $D/log_s${S}_K${K}_c{}.txt"
