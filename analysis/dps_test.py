"""Diffusion assimilation (DPS, bg_obs) and persistence-free forecast (free_run) on the 600-frame test cases, evaluation frames 400..599. The cycle starts at frame START (default 0, as the EnKF): frame START is reconstructed from its observations only, then the solver forecasts and data re-enter every K frames.
python analysis/dps_test.py DEVICE OUTDIR JOB [JOB ...]    JOB = seed:case ; env KS='30,10,0,1' (0 = free_run, independent of K), SPLIT=test, BG_STD=0.3 (background std of the DPS likelihood term; larger = weaker background), START=0, FIRST=20 (other-init only: no data before frame FIRST, then every K frames), INIT=frame0|other (other: cycle-0 background = last of 20 solver-integrated frames from the EnKF donor, no frame-0 reconstruction; free forecast starts from it and never assimilates), WARMUP=0 (solver-only frames after the frame-0 reconstruction)"""
import os, sys, glob, time, shutil, tempfile, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.rb_noise import rb_nodes
from rdda.solver.no_slip_projection import project_field
dev, out = sys.argv[1], Path(sys.argv[2]); jobs = [tuple(int(v) for v in j.split(':')) for j in sys.argv[3:]]; out.mkdir(parents=True, exist_ok=True)
KS = [int(k) for k in os.environ.get('KS', '30,10,0,1').split(',')]; split = os.environ.get('SPLIT', 'test'); START = int(os.environ.get('START', 0)); WARMUP = int(os.environ.get('WARMUP', 0)); INIT = os.environ.get('INIT', 'frame0'); FIRST = int(os.environ['FIRST']) if os.environ.get('FIRST') else None; END = 599; train = sorted(glob.glob('data/processed/train/case_*.npy')); z = rb_nodes(S.NZ, S.LZ)
for K in KS:
    for seed, case in jobs:
        f = out / (f'free_s{seed}_c{case}.csv' if K == 0 else f's{seed}_K{K}_c{case}.csv')
        if f.exists(): continue
        a = argparse.Namespace(data='data/processed_600', split=split, case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=START, analysis_steps=END - START, rayleigh=1e5, model_rayleigh=5e4,
            observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=float(os.environ.get('BG_STD', .3)), dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
            obs_noise=.1, every=max(K, 1), seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
        truth, mask, fc = S.setup(a)
        init = None
        if INIT == 'other':   # same donor frame as the EnKF ensemble; the solver integrates 20 consecutive frames and the LAST one is the background of cycle 0
            rng = np.random.default_rng(case); donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
            x = fc.consistent(np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32))
            if FIRST is None:
                for _ in range(19): x = fc(x, fc.steps_per_frame)[0]   # old setting: assimilate at frame 0 with the 20th integrated state
            init = x                                                  # FIRST=20: frame 0 IS the donor frame, frames 1..FIRST-1 are solver integration, data enter at frame FIRST
        t0 = time.perf_counter(); tmp = tempfile.mkdtemp(dir='/tmp')
        rows = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=('no_assim' if init is not None else 'free_run') if K == 0 else 'bg_obs', assim_every=max(K, 1), warmup=WARMUP, first_assim=FIRST, init_state=init, out=tmp, **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
        t = time.perf_counter() - t0
        df = pd.DataFrame(rows)[['frame', 'assimilated', 'missing_rmse']].assign(frame=lambda d: d.frame + START, K=K, seed=seed, case=case, t_run=t, n_assim=sum(r['assimilated'] for r in rows)); df = df[df.frame >= 400].reset_index(drop=True)
        an = np.stack([np.load(f'{tmp}/frame_{r - START:04d}.npz')['analysis'] for r in df.frame])       # analyses of frames 400..599, saved for later checks
        df['an_div_rel'] = [fc.balance(x)['div_rel'] for x in an]; df['proj_div_rel'] = [fc.balance(project_field(x, z, S.LX, S.LZ))['div_rel'] for x in an]
        df['truth_div_rel'] = [fc.balance(truth[r - START])['div_rel'] for r in df.frame]
        np.savez_compressed(str(f.with_suffix('')) + '_analysis.npz', frames=df.frame.values, analysis=an); shutil.rmtree(tmp)
        df.to_csv(str(f) + '.tmp', index=False); os.replace(str(f) + '.tmp', f)
        print(f'{"free" if K == 0 else "DPS"} K={K} seed={seed} case={case} rmse(400-599)={df.missing_rmse.mean():.4f} time={t:.0f}s', flush=True)
