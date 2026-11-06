#!/bin/bash
cd /data1/user/lz/RB_Diffusion_DA; export OMP_NUM_THREADS=1
s=$1; [ -f data/raw_720/seed_$s/case_00000.npy ] && exit 0
~/miniconda3/envs/dedalus3/bin/python -m rdda.solver.generate_data --out data/raw_720/seed_$s --cases 1 --steps 25200 --seed-start $s 2>&1 | grep -v UCX > data/raw_720/logs/seed_$s.log
