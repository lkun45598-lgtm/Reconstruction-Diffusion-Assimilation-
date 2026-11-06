"""Run one test case (600-frame data) with diffusion DA / EnKF (other-case init) / free forecast and SAVE frame_XXXX.npz (truth, analysis, mask, observations) for analysis/make_rb_gif.py.
python analysis/save_case_frames.py MODE DEVICE SEED K CASE OUTDIR      MODE: dps | free | enkf  (dps/free: frames 400..599 saved as frame_0000..0199; enkf: frames 400..599 of a run from frame 0 saved as frame_0000..0199; INFL env = inflation)"""
import os, sys, glob, shutil, argparse, numpy as np
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
mode, dev, seed, K, case, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), Path(sys.argv[6]); out.mkdir(parents=True, exist_ok=True)
a = argparse.Namespace(data='data/processed_600', split='test', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=400 if mode != 'enkf' else 0, analysis_steps=199 if mode != 'enkf' else 599, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
    obs_noise=.1, every=max(K, 1), seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a)
if mode == 'enkf':
    train = sorted(glob.glob('data/processed/train/case_*.npy')); rng = np.random.default_rng(case); donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
    x = np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32); ens0 = [fc.consistent(x)]
    for _ in range(19): ens0.append(fc(ens0[-1], fc.steps_per_frame)[0])
    kw = S.enkf_kw(a, np.zeros_like(truth[0])); kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=float(os.environ['INFL']), loc_radius=None, assim_every=K)
    tmp = out / 'full'; tmp.mkdir(exist_ok=True); rows = run_enkf_cycle(truth, fc, mask, out=str(tmp), **kw)
    for k in range(200): shutil.copy(tmp / f'frame_{400 + k:04d}.npz', out / f'frame_{k:04d}.npz')
    shutil.rmtree(tmp)
else:
    rows = run_cycle(truth, a.prior, fc, mask, analysis_steps=199, method='free_run' if mode == 'free' else 'bg_obs', assim_every=max(K, 1), out=str(out), **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
print(mode, 'done', np.mean([r['missing_rmse'] for r in rows[-200:]]))
