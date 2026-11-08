#!/bin/bash
# after the validation selection: frames + gifs of test case 0 (seed 42) with the selected parameters, new setting (FIRST=20)
cd /data1/user/lz/RB_Diffusion_DA; P=~/miniconda3/envs/dedalus3/bin/python; SW=outputs/261107_sweep_first20; R=outputs/261107_gif_first20_final_case0; mkdir -p $R/K10 $R/K30 $R/free
until [ -f $SW/selected.json ]; do sleep 30; done
read BG10 BG30 I10 I30 < <($P -c "import json;s=json.load(open('$SW/selected.json'));print(s['bg10'],s['bg30'],s['infl10'],s['infl30'])")
ln -sfn ../free $R/K10/free; ln -sfn ../free $R/K30/free; E="OMP_NUM_THREADS=1 FULL=1 INIT=other FIRST=20"
( env CUDA_VISIBLE_DEVICES=3 $E BG_STD=$BG10 $P analysis/save_case_frames.py dps cuda:0 42 10 0 $R/K10/dps 2>&1 | grep -v UCX > $R/log_dps10.txt ) &
( env CUDA_VISIBLE_DEVICES=7 $E BG_STD=$BG30 $P analysis/save_case_frames.py dps cuda:0 42 30 0 $R/K30/dps 2>&1 | grep -v UCX > $R/log_dps30.txt ) &
( env CUDA_VISIBLE_DEVICES=7 $E $P analysis/save_case_frames.py free cuda:0 42 10 0 $R/free 2>&1 | grep -v UCX > $R/log_free.txt ) &
( env $E RDDA_FORECAST_PROCS=10 INFL=$I10 $P analysis/save_case_frames.py enkf cuda:0 42 10 0 $R/K10/enkf 2>&1 | grep -v UCX > $R/log_enkf10.txt ) &
( env $E RDDA_FORECAST_PROCS=10 INFL=$I30 $P analysis/save_case_frames.py enkf cuda:0 42 30 0 $R/K30/enkf 2>&1 | grep -v UCX > $R/log_enkf30.txt ) &
wait
NOTE=$'All three start from the same donor state (a frame of a training case; EnKF uses 20 consecutive frames of it as its members). Frames 0-19 are pure solver integration.\nDiffusion DA and EnKF assimilate from frame 20, then every K frames; the free forecast never assimilates.'
for K in 10 30; do mkdir -p $R/K$K/gif; i=0; for n in b ux uz; do $P analysis/make_rb_gif.py --arms "Diffusion DA=$R/K$K/dps" "EnKF=$R/K$K/enkf" "Free forecast=$R/K$K/free" --channel $i --every $K --stride 2 --note "$NOTE" --out $R/K$K/gif/rb_$n > $R/K$K/gif/log_$n.txt 2>&1 & i=$((i+1)); done; done; wait
echo GIF_DONE > $R/done.txt
