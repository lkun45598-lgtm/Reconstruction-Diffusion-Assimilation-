#!/bin/bash
# new setting (frames 0-19 pure solver integration from the EnKF donor, first assimilation at frame 20): wait for the validation sweeps -> select params -> test set (DPS, free, EnKF; 20 cases x seeds 42,43,44; analysis + divergence saved) -> gif frames/gifs of test case 0
cd /data1/user/lz/RB_Diffusion_DA; P=~/miniconda3/envs/dedalus3/bin/python; SW=outputs/261107_sweep_first20; T=outputs/261107_first20_test; mkdir -p $T
while tmux has-session -t swf20 2>/dev/null || tmux has-session -t swenkf 2>/dev/null; do sleep 30; done
$P analysis/select_params_first20.py $SW | tee $SW/selection.txt
read BG10 BG30 I10 I30 < <($P -c "import json;s=json.load(open('$SW/selected.json'));print(s['bg10'],s['bg30'],s['infl10'],s['infl30'])")
echo "selected: bg10=$BG10 bg30=$BG30 infl10=$I10 infl30=$I30" | tee $T/params.txt
# DPS + free (GPUs 2 3 6 7, retried)
GL=(2 3 6 7); n=0
( for spec in "10 $BG10" "30 $BG30" "0 $BG10"; do set -- $spec; for c in $(seq 0 19); do g=${GL[$((n % 4))]}; n=$((n+1))
  echo "for t in 1 2 3; do CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=1 INIT=other FIRST=20 SPLIT=test BG_STD=$2 KS=$1 START=0 $P analysis/dps_test.py cuda:0 $T/dps 42:$c 43:$c 44:$c 2>&1 | grep -v UCX >> $T/log_dps_K$1_c$c.txt; done"; done; done | xargs -P 8 -I{} bash -c "{}" ) &
# EnKF (single thread each, SAVE=1)
( for K in 10 30; do I=$([ $K = 10 ] && echo $I10 || echo $I30); for seed in 42 43 44; do for c in $(seq 0 19); do
  echo "OMP_NUM_THREADS=1 STEPS=599 KS=$K INFL=$I FIRST=20 SAVE=1 $P analysis/other_case_init.py test $seed $T/enkf $c 2>&1 | grep -v UCX > $T/log_enkf_K${K}_s${seed}_c$c.txt"; done; done; done | xargs -P 60 -I{} bash -c "{}" ) &
wait; echo ALL_TEST_DONE | tee $T/done.txt
