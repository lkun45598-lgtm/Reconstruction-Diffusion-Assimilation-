"""Relative divergence (raw / after the common no-slip projection) of the analyses on frames 400..599 of the 600-frame test cases, same runs as other_case_init.py (EnKF) and dps_test.py (DPS, free forecast).
python analysis/div_eval.py MODE DEVICE SEED OUTDIR K CASE [CASE ...]     MODE: dps | enkf     (K=0 in dps mode = free forecast). Saved analyses are deleted after the divergence is computed."""
import os, sys, glob, shutil, tempfile, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
from rdda.rb_noise import rb_nodes
from rdda.solver.no_slip_projection import project_field
mode, dev, seed, out, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4]), int(sys.argv[5]); cases = [int(c) for c in sys.argv[6:]]; out.mkdir(parents=True, exist_ok=True); z = rb_nodes(S.NZ, S.LZ)
train = sorted(glob.glob('data/processed/train/case_*.npy')); INFL = float(os.environ.get('INFL', 1.1))
for case in cases:
    f = out / f'{mode}_s{seed}_K{K}_c{case}.csv'
    if f.exists(): continue
    a = argparse.Namespace(data='data/processed_600', split='test', case_index=case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=400 if mode == 'dps' else 0, analysis_steps=199 if mode == 'dps' else 599, rayleigh=1e5, model_rayleigh=5e4,
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=max(K, 1), seed=seed, device=dev, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a); tmp = tempfile.mkdtemp(dir='/tmp'); off = a.start_frame
    if mode == 'dps':
        rows = run_cycle(truth, a.prior, fc, mask, analysis_steps=199, method='free_run' if K == 0 else 'bg_obs', assim_every=max(K, 1), out=tmp, **S.cyc_kw(a), **S.dps_kw(a, project_output=True))
    else:
        rng = np.random.default_rng(0 * 100003 + case); donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
        x = np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32); ens0 = [fc.consistent(x)]
        for _ in range(19): ens0.append(fc(ens0[-1], fc.steps_per_frame)[0])
        kw = S.enkf_kw(a, np.zeros_like(truth[0])); kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=INFL, loc_radius=None, assim_every=K)
        rows = run_enkf_cycle(truth, fc, mask, out=tmp, **kw)
    res = []
    for r in rows:
        fr_ = r['frame'] + off
        if fr_ < 400: continue
        an = np.load(f'{tmp}/frame_{r["frame"]:04d}.npz')['analysis']
        res.append(dict(mode=mode, K=K, seed=seed, case=case, frame=fr_, missing_rmse=r['missing_rmse'], an_div_rel=fc.balance(an)['div_rel'], proj_div_rel=fc.balance(project_field(an, z, S.LX, S.LZ))['div_rel'], truth_div_rel=fc.balance(truth[r['frame']])['div_rel']))
    shutil.rmtree(tmp); d = pd.DataFrame(res); d.to_csv(str(f) + '.tmp', index=False); os.replace(str(f) + '.tmp', f)
    print(mode, K, seed, case, 'rmse %.4f raw %.2e proj %.2e truth %.2e' % (d.missing_rmse.mean(), d.an_div_rel.mean(), d.proj_div_rel.mean(), d.truth_div_rel.mean()), flush=True)
