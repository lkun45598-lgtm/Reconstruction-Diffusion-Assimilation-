"""EnKF whose initial ensemble comes from ANOTHER case (no information of the assimilated case): a random train case, a random frame of it,
integrated by the forecast solver (Ra_model) for 20 consecutive frames -> 20 members (members stay parallel trajectories afterwards, standard EnKF).
Assimilation starts at frame 0 of the case (skip_t0=False); the per-frame missing_rmse of every frame is saved (evaluate on frames >= 40 later). Wall time is recorded.
python analysis/other_case_init.py SPLIT SEED OUTDIR CASE [CASE ...]
env: KS='1,20'; INFL='1.1' (inflation, one value or 'K:v,K:v'); DONOR_SEED=0 (random choice of donor case/frame)"""
import os, sys, time, glob, shutil, tempfile, argparse, numpy as np, pandas as pd
sys.path.insert(0, 'analysis'); import study_rb as S
from pathlib import Path
from rdda.enkf import run_enkf_cycle
from rdda.rb_noise import rb_nodes
from rdda.solver.no_slip_projection import project_field
split, seed, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]); cases = [int(c) for c in sys.argv[4:]]
KS = [int(k) for k in os.environ.get('KS', '1,20').split(',')]; out.mkdir(parents=True, exist_ok=True)
INFL = {k: float(os.environ.get('INFL', '1.1')) for k in KS}
if ':' in os.environ.get('INFL', ''): INFL = {int(a): float(b) for a, b in (kv.split(':') for kv in os.environ['INFL'].split(','))}
train = sorted(glob.glob(os.environ.get('DONOR_DIR', 'data/processed/train') + '/case_*.npy'))   # donor pool: env DONOR_DIR (default: the Ra=1e5 training trajectories)
for case in cases:
    a = argparse.Namespace(data=os.environ.get('DATA', 'data/processed_600'), split=split, case_index=case, prior='', out='x', start_frame=0, analysis_steps=int(os.environ.get('STEPS', 599)), rayleigh=1e5, model_rayleigh=float(os.environ.get('MRA', 5e4)), prandtl=float(os.environ.get('PR', 1.0)),
        observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=.3, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
        obs_noise=.1, every=1, seed=seed, device='cpu', enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
    truth, mask, fc = S.setup(a)
    rng = np.random.default_rng(int(os.environ.get('DONOR_SEED', 0)) * 100003 + case)
    donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
    t0 = time.perf_counter(); x = np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32); ens0 = [fc.consistent(x)]
    FIRST = int(os.environ['FIRST']) if os.environ.get('FIRST') else None
    for _ in range(FIRST if FIRST else 19): ens0.append(fc(ens0[-1], fc.steps_per_frame)[0])
    if FIRST: pre_chain, ens0 = ens0[:FIRST], ens0[1:FIRST + 1]       # FIRST=20: chain states 0..20; members = states 1..20 (the ensemble at frame 20), frames 0..19 shown as the chain
    t_init = time.perf_counter() - t0
    for K in KS:
        f = out / f's{seed}_K{K}_c{case}.csv'
        if f.exists(): continue
        a.every = K; kw = S.enkf_kw(a, np.zeros_like(truth[0]))
        kw.update(init_center=None, init_ensemble=ens0, skip_t0=False, inflation=INFL[K], loc_radius=None, assim_every=K, first_assim=FIRST, **({'ens_at_first': True, 'pre_chain': pre_chain} if FIRST else {}))
        t0 = time.perf_counter(); SAVE = os.environ.get('SAVE') == '1'; tmp = tempfile.mkdtemp(dir='/tmp') if SAVE else None; rows = run_enkf_cycle(truth, fc, mask, out=tmp, **kw); t_run = time.perf_counter() - t0
        n_assim = sum(r['assimilated'] for r in rows)
        df = pd.DataFrame(rows)[['frame', 'assimilated', 'missing_rmse', 'ensemble_spread']]
        df = df.assign(K=K, seed=seed, case=case, donor=donor, donor_frame=fr, t_init=t_init, t_run=t_run, n_assim=n_assim, infl=INFL[K])
        if SAVE:   # SAVE=1: divergence of the analysed mean field (raw / after the common no-slip projection) and the analyses of frames 400..599
            z = rb_nodes(S.NZ, S.LZ); fr_ = [t for t in range(len(df)) if t >= int(os.environ.get('SAVE_FROM', 400))]
            an = np.stack([np.load(f'{tmp}/frame_{t:04d}.npz')['analysis'] for t in fr_])
            df['an_div_rel'] = np.nan; df['proj_div_rel'] = np.nan; df['truth_div_rel'] = np.nan
            df.loc[fr_, 'an_div_rel'] = [fc.balance(x)['div_rel'] for x in an]; df.loc[fr_, 'proj_div_rel'] = [fc.balance(project_field(x, z, S.LX, S.LZ))['div_rel'] for x in an]
            df.loc[fr_, 'truth_div_rel'] = [fc.balance(truth[t])['div_rel'] for t in fr_]
            np.savez_compressed(str(f.with_suffix('')) + '_analysis.npz', frames=np.array(fr_), analysis=an); shutil.rmtree(tmp)
        df.to_csv(f, index=False)
        e = df[df.frame >= 40].missing_rmse.mean()
        print(f'{split} c{case} K{K} donor{donor}@{fr} rmse(f0)={df.missing_rmse[0]:.4f} rmse(f>=40)={e:.4f} spread0={df.ensemble_spread[0]:.4f} time={t_run:.0f}s ({t_run / len(df):.2f}s/frame)', flush=True)
