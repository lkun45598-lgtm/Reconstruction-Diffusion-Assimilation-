"""RB studies (mirror of study_ns2d.py): (A) initial-condition noise, (B) EnKF vs diffusion-DPS (+ the sweeps that choose
their parameters on a validation case), (C) single-frame reconstruction with / without a background field.

State = (b, u_x, u_z) on 72x24 (x periodic, z Chebyshev-Gauss nodes between the walls). Forecast model = Dedalus with a
mismatched Rayleigh number (--model-rayleigh; truth Ra=1e5). One saved frame = 0.35 time units = 35 solver steps.
Errors: RMSE at UNOBSERVED points, pooled over the 3 channels (`missing_rmse`), per channel (`missing_rmse_ch*`) and
normalised by the training channel stds (`missing_nrmse`). Balance = relative divergence (spectral, rdda.diagnostics).
Noise: rdda.rb_noise (b' wall-vanishing, u' from a stream function => divergence free). obs_std/bg_std are physical
units here; observations are noise-free truth, so obs_std is only the likelihood weight / EnKF R, not a noise level.
Sweeps run on a VALIDATION case, the comparison on a TEST case. One process per sample/config: use `merge` afterwards.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from rdda.assimilation import assimilate, make_mask, metric, noisy_observations
from rdda.cycle import run_cycle
from rdda.enkf import run_enkf_cycle
from rdda.forecasters import RBForecaster
from rdda.rb_noise import perturbation, rb_nodes, reference_std
from rdda.solver.rb_equations import RBConfig

NX, NZ, LX, LZ = 72, 24, 3.0, 1.0


def common(p):
    p.add_argument('--data', default='data/processed')
    p.add_argument('--split', default='test', help='test for comparisons, val for parameter sweeps')
    p.add_argument('--case-index', type=int, default=0)
    p.add_argument('--prior', default='outputs/rb_prior_4gpu/best.pt')
    p.add_argument('--out', required=True)
    p.add_argument('--start-frame', type=int, default=0)
    p.add_argument('--analysis-steps', type=int, default=99)
    p.add_argument('--rayleigh', type=float, default=1e5)
    p.add_argument('--model-rayleigh', type=float, default=5e4)
    p.add_argument('--observation-fraction', type=float, default=0.01)
    p.add_argument('--members', type=int, default=8, help='DPS samples per analysis')
    p.add_argument('--dps-steps', type=int, default=100)
    p.add_argument('--obs-std', type=float, default=0.05)
    p.add_argument('--bg-std', type=float, default=0.3)
    p.add_argument('--dps-rate', type=float, default=0.5)
    p.add_argument('--init', default='noise', choices=['noise', 'background'])
    p.add_argument('--t-start', type=float, default=0.4)
    p.add_argument('--hard-observation', action=argparse.BooleanOptionalAction, default=True,
                   help='overwrite observed points in the analysis (DPS) / shift the EnKF mean there (EnKF): keep equal')
    p.add_argument('--div-weight', type=float, default=0.0, help='incompressibility guidance weight in DPS (0 = off)')
    p.add_argument('--obs-noise', type=float, default=0.0, help='std of the Gaussian error added to the observations (physical units; 0 = exact)')
    p.add_argument('--every', type=int, default=3, help='observations enter every K frames (3 = 1.05 time units)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')


DIV_SCALE = 1.0      # rms |grad u| of a reference field; set in setup()


def setup(a):
    f = sorted((Path(a.data) / a.split).glob('case_*.npy'))[a.case_index]
    truth = np.load(f)[a.start_frame:].astype(np.float32)
    mask = make_mask(truth.shape[1:], a.observation_fraction, a.seed)
    fc = RBForecaster(RBConfig(nx=NX, nz=NZ, lx=LX, lz=LZ, rayleigh=a.model_rayleigh, prandtl=1.0, dt=0.01), 'streamfunction')
    a.std = np.load(Path(a.data) / 'stats.npz')['std']
    global DIV_SCALE
    DIV_SCALE = rms_grad(truth[0])
    print('case', f, 'obs pixels', int(mask.any(-1).sum()), flush=True)
    return truth, mask, fc


def rms_grad(x):
    """rms of |grad u| of a reference state: the unit in which the divergence penalty is measured."""
    from rdda.diagnostics import _spectral_d, _cheb_dz
    ux, uz = x[..., 1].astype(np.float64), x[..., 2].astype(np.float64); z = rb_nodes(NZ, LZ)
    return float(np.sqrt(np.mean(_spectral_d(ux, 0, LX) ** 2 + _cheb_dz(ux, z, LZ) ** 2 + _spectral_d(uz, 0, LX) ** 2 + _cheb_dz(uz, z, LZ) ** 2)))


def dps_kw(a, **over):
    kw = dict(members=a.members, steps=a.dps_steps, obs_std=a.obs_std, bg_std=a.bg_std, relative_std=False, seed=a.seed,
              device=a.device, hard_observation=a.hard_observation, dps_rate=a.dps_rate,
              div_weight=a.div_weight, div_scale=DIV_SCALE, div_geometry=(rb_nodes(NZ, LZ), LX, LZ))
    kw.update(over)
    return kw


def cyc_kw(a):
    return dict(obs_noise=a.obs_noise, obs_seed=a.seed)


def nrmse(r, std):
    return float(np.sqrt(np.mean([(r[f'missing_rmse_ch{k}'] / std[k]) ** 2 for k in range(3)])))


def enrich(rows, d, fc, truth, std):
    """Add balance of the saved analysis, balance of the truth, and the normalised error to every row."""
    for r in rows:
        an = np.load(Path(d) / f'frame_{r["frame"]:04d}.npz')['analysis']
        r.update({f'an_{k}': v for k, v in fc.balance(an).items() if v is not None})
        r['truth_div_rel'] = fc.balance(truth[r['truth_frame']])['div_rel']
        r['missing_nrmse'] = nrmse(r, std)
    return rows


def dps_cycle(a, truth, mask, fc, out, *, method='bg_obs', every=None, init_state=None, **over):
    return run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=method,
                     assim_every=a.every if every is None else every, init_state=init_state, out=out, **cyc_kw(a),
                     **dps_kw(a, init=a.init, t_start=a.t_start, **over) if a.init == 'background' else dps_kw(a, **over))


def enkf_kw(a, center, ref_state=None):
    """RB-specific EnKF options: physical-coordinate localization, consistent initial perturbation, no low-pass."""
    ref = reference_std(center if ref_state is None else ref_state)
    return dict(members=a.enkf_members, obs_std=getattr(a, 'enkf_obs_std', a.obs_std), relative_std=False, std_ref=a.std, init_center=center,
                analysis_steps=a.analysis_steps, assim_every=a.every, seed=a.seed, hard_observation=a.hard_observation, obs_noise=a.obs_noise, obs_seed=a.seed, demean_init=getattr(a, 'demean_init', False),
                coords=(np.arange(NX) * LX / NX, rb_nodes(NZ, LZ)), lengths=(LX, None), lowpass_k=None,
                perturb=lambda r: perturbation(r, ref, a.init_level)[0])


# ------------------------------------------------------------------------------------------------ A. noise
def noise(a):
    """(A) perturb the exact initial state (all 3 channels, physically consistent), compare no assimilation with DPS.
    Both start from the same noisy first guess. level = fraction of the reference std (b: with the x-mean removed)."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ref = reference_std(truth[0])
    print('reference std (b - <b>_x, u_x, u_z):', ref, flush=True)
    arms = [('No assimilation (forecast only)', 'no_assim', 1),
            (f'DPS, observations every {a.every} frames', 'bg_obs', a.every)]
    rows = []
    for sid in a.samples:
        for level in a.levels:
            pert, realised = perturbation(np.random.default_rng(a.seed * 100 + sid), ref, level)
            x0 = (truth[0] + pert).astype(np.float32)
            proj = float(fc(x0, 1)[1]['initial_projection_rmse'])          # how much the Dedalus restart alters x0
            meta = dict(level=level, sample=sid, init_rmse=metric(x0, truth[0]), init_restart_change=proj,
                        noise_std_b=float(realised[0]), noise_std_ux=float(realised[1]), noise_std_uz=float(realised[2]))
            schemes = list(arms) + ([(f'DPS, observations every frame', 'bg_obs', 1)] if sid in a.control_samples else [])
            for name, method, every in schemes:
                d = out / f'noise{level}_s{sid}_{method}_{every}'; d.mkdir(exist_ok=True)
                res = dps_cycle(a, truth, mask, fc, d, method=method, every=every, init_state=x0)
                rows += [dict(scheme=name, every=every, **meta, **r) for r in enrich(res, d, fc, truth, a.std)]
                pd.DataFrame(rows).to_csv(out / 'noise.csv', index=False)
    print('saved', out / 'noise.csv')


# ------------------------------------------------------------------------------------------------ B. EnKF / DPS
def center_analysis(a, truth, mask):
    """Frame-0 DPS analysis from the observations only (shared start of the DPS-vs-EnKF comparison)."""
    return assimilate(a.prior, noisy_observations(truth[0], mask, a.obs_noise, a.seed, 0), mask, **dps_kw(a))['mean']


def enkf(a):
    """(B) tuned DPS vs EnKF variants, same observations / forecast model / hard-observation policy.
    Start: the EnKF ensemble is centred on DPS's frame-0 analysis (and skips its own t=0 update), so this compares the
    cycling AFTER a common first analysis, not two independent initialisations. EnKF balance = before the restart."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, center = [], None
    for name in a.schemes:
        d = out / name; d.mkdir(exist_ok=True)
        if name == 'dps':
            res = dps_cycle(a, truth, mask, fc, d)
            center = np.load(d / 'frame_0000.npz')['analysis']
        else:
            kw = enkf_kw(a, center)
            kw.update(inflation=1.0 if name == 'enkf_plain' else a.inflation,
                      loc_radius=(a.radius_x, a.radius_z) if name == 'enkf_infl_loc' else None)
            res = run_enkf_cycle(truth, fc, mask, out=str(d), **kw)
        rows += [dict(scheme=name, **r) for r in enrich(res, d, fc, truth, a.std)]
        pd.DataFrame(rows).to_csv(out / 'enkf.csv', index=False)
    print('saved', out / 'enkf.csv')


def enkf_sweep(a):
    """Choose EnKF inflation x localization radii (physical length: x periodic length 3, z wall-to-wall length 1)
    on the validation case; score = mean unobserved RMSE."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.center_file and Path(a.center_file).exists(): center = np.load(a.center_file)
    else:
        center = center_analysis(a, truth, mask)
        if a.center_file: np.save(a.center_file, center)
    rows = []
    for infl in a.inflations:
        for rx in a.radii_x:
            for rz in a.radii_z:
                kw = enkf_kw(a, center)
                kw.update(inflation=infl, loc_radius=(rx, rz) if rx > 0 else None)
                try:
                    res = run_enkf_cycle(truth, fc, mask, **kw)
                    e = [r['missing_rmse'] for r in res]
                    rows.append(dict(inflation=infl, radius_x=rx, radius_z=rz, members=a.enkf_members, mean_rmse=float(np.mean(e)),
                                     final_rmse=e[-1], mean_nrmse=float(np.mean([nrmse(r, a.std) for r in res])), blew_up=False))
                except (FloatingPointError, ValueError) as err:           # a member's forecast diverged: the filter is unstable here
                    rows.append(dict(inflation=infl, radius_x=rx, radius_z=rz, members=a.enkf_members, mean_rmse=float('nan'),
                                     final_rmse=float('nan'), mean_nrmse=float('nan'), blew_up=True))
                    print('blew up:', err, flush=True)
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out / 'enkf_sweep.csv', index=False)


def dps_sweep(a):
    """Tune DPS on the validation case; config 'bg_std:dps_rate:obs_std:init[:t_start]'; score = mean unobserved RMSE."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in a.grid:
        f = cfg.split(':')
        over = dict(bg_std=float(f[0]), dps_rate=float(f[1]), obs_std=float(f[2]), init=f[3])
        if f[3] == 'background': over['t_start'] = float(f[4])
        try:
            res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=a.method, assim_every=a.every,
                            **cyc_kw(a), **dps_kw(a, **over))
            e = [r['missing_rmse'] for r in res]
            rows.append(dict(config=cfg, mean_rmse=float(np.mean(e)), final_rmse=e[-1],
                             mean_nrmse=float(np.mean([nrmse(r, a.std) for r in res]))))
        except FloatingPointError:                                  # DPS diverged for this setting
            rows.append(dict(config=cfg, mean_rmse=float('nan'), final_rmse=float('nan'), mean_nrmse=float('nan')))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(out / 'dps_sweep.csv', index=False)


# ------------------------------------------------------------------------------------------------ C. reconstruction
def recon(a):
    """(C) single-frame reconstruction, with / without a background. Backgrounds: (smooth) truth + consistent noise of
    relative size sigma (sigma=0 is the exact-background UPPER BOUND), and (model) the Ra-mismatched model run from the
    truth `lead` frames earlier (a real model error)."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fcs = {ra: RBForecaster(RBConfig(nx=NX, nz=NZ, lx=LX, lz=LZ, rayleigh=ra, prandtl=1.0, dt=0.01), 'streamfunction') for ra in a.model_rayleighs}
    schemes = (('no background (obs only)', dict(background=None, init='noise', obs_std=a.obsonly_obs_std, dps_rate=a.obsonly_rate)),
               ('background as guidance', dict(init='noise', bg_std=a.guidance_bg_std, obs_std=a.guidance_obs_std,
                                               dps_rate=a.guidance_rate)),
               ('start from noised background', dict(init='background', t_start=a.t_start, bg_guidance=False)))
    rows = []
    for f in a.frames:
        xt = truth[f]
        obs = np.where(mask, xt, 0)
        ref = reference_std(xt)
        bgs = []
        for sigma in a.bg_errors:                                   # optional artificial smooth error (off by default)
            bgs.append(('smooth', sigma, np.nan, xt if sigma == 0 else (xt + perturbation(np.random.default_rng(a.seed + f), ref, sigma)[0])))
        for ra in a.model_rayleighs:                                # real model error: mismatched-Ra Dedalus run from the truth
            for lead in a.model_leads:
                if f - lead >= 0: bgs.append(('model', lead, ra, fcs[ra](truth[f - lead], lead * fc.steps_per_frame)[0]))
        for kind, par, ra, bg in bgs:
            for label, kw in schemes:
                kw = dict(kw)
                if label != 'no background (obs only)': kw['background'] = bg
                r = assimilate(a.prior, obs, mask, **dps_kw(a, seed=a.seed + f, **kw))
                m = r['mean']
                row = dict(frame=f, bg_kind=kind, bg_param=par, bg_ra=ra, scheme=label,
                           bg_missing_rmse=metric(bg, xt, ~mask), missing_rmse=metric(m, xt, ~mask),
                           **{f'bg_missing_rmse_ch{k}': metric(bg[..., k:k + 1], xt[..., k:k + 1], (~mask)[..., k:k + 1]) for k in range(3)},
                           **{f'missing_rmse_ch{k}': metric(m[..., k:k + 1], xt[..., k:k + 1], (~mask)[..., k:k + 1]) for k in range(3)},
                           **{f'an_{k}': v for k, v in fc.balance(m).items() if v is not None}, truth_div_rel=fc.balance(xt)['div_rel'])
                row['missing_nrmse'] = nrmse(row, a.std)
                row['bg_nrmse'] = float(np.sqrt(np.mean([(row[f'bg_missing_rmse_ch{k}'] / a.std[k]) ** 2 for k in range(3)])))
                rows.append(row)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in ('frame', 'bg_kind', 'bg_param', 'scheme', 'missing_rmse', 'bg_missing_rmse')}, flush=True)
        pd.DataFrame(rows).to_csv(out / 'recon.csv', index=False)
    print('saved', out / 'recon.csv')


def fusion(a):
    """(D) with vs without the numerical background, observations in EVERY frame (the cycle itself):
    bg_obs   = background (Dedalus forecast of the previous analysis, Ra mismatched) + this frame's observations -> DPS
    obs_only = this frame's observations only -> DPS (no solver). Same mask, same prior, each arm tuned on the validation case."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arms = (('DPS with background (solver forecast + observations)', 'bg_obs', {}),
            ('DPS with observations only (no background)', 'obs_only',
             dict(init='noise', obs_std=a.obsonly_obs_std, dps_rate=a.obsonly_rate)))
    rows = []
    for name, method, over in arms:
        d = out / method; d.mkdir(exist_ok=True)
        kw = dps_kw(a, **over)
        if method == 'bg_obs' and a.init == 'background': kw.update(init=a.init, t_start=a.t_start)
        res = run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method=method, assim_every=1, out=d, **cyc_kw(a), **kw)
        rows += [dict(scheme=name, case=a.case_index, **r) for r in enrich(res, d, fc, truth, a.std)]
        pd.DataFrame(rows).to_csv(out / 'fusion.csv', index=False)
    print('saved', out / 'fusion.csv')


# ------------------------------------------------------------------------------------------------ E. bad first guess
def init_state_of(spec, truth):
    """'zero' | 'scale:s' (s x truth[0]) | 'later:K' (truth[K], a state of a different phase of the same flow)."""
    kind, _, v = spec.partition(':')
    return {'zero': lambda: np.zeros_like(truth[0]), 'scale': lambda: float(v) * truth[0], 'later': lambda: truth[int(v)].copy()}[kind]()


def spinup(a):
    """(E) start far from the truth (zero field / weakened field / a state of another time), noisy observations every frame:
    can the observations pull each scheme back?  A = DPS+background, C = EnKF (ensemble centred on the bad guess, t=0 analysed),
    B = DPS observations only (does not depend on the first guess), F = free forecast from the bad guess."""
    truth, mask, fc = setup(a)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(name, init, d, res):
        rows.extend(dict(scheme=name, init=init, **r) for r in enrich(res, d, fc, truth, a.std))
        pd.DataFrame(rows).to_csv(out / 'spinup.csv', index=False)

    d = out / 'obs_only'; d.mkdir(exist_ok=True)
    add('B', 'any', d, run_cycle(truth, a.prior, fc, mask, analysis_steps=a.analysis_steps, method='obs_only', assim_every=1, out=d, **cyc_kw(a),
                                 **dps_kw(a, init='noise', obs_std=a.obsonly_obs_std, dps_rate=a.obsonly_rate)))
    for spec in a.inits:
        x0 = init_state_of(spec, truth).astype(np.float32); tag = spec.replace(':', '')
        print('init', spec, 'rmse of the first guess', metric(x0, truth[0]), flush=True)
        d = out / f'{tag}_free'; d.mkdir(exist_ok=True)
        add('F', spec, d, dps_cycle(a, truth, mask, fc, d, method='no_assim', every=1, init_state=x0))
        d = out / f'{tag}_A'; d.mkdir(exist_ok=True)
        add('A', spec, d, dps_cycle(a, truth, mask, fc, d, every=1, init_state=x0))
        d = out / f'{tag}_C'; d.mkdir(exist_ok=True)
        kw = enkf_kw(a, x0, ref_state=truth[0]); kw.update(inflation=a.inflation, loc_radius=(a.radius_x, a.radius_z), skip_t0=False, assim_every=1)
        add('C', spec, d, run_enkf_cycle(truth, fc, mask, out=str(d), **kw))
    print('saved', out / 'spinup.csv')


def merge(a):
    pd.concat([pd.read_csv(f) for f in a.inputs]).to_csv(a.output, index=False)
    print('saved', a.output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('noise'); common(n)
    n.add_argument('--levels', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.2, 0.4], help='x reference std')
    n.add_argument('--samples', type=int, nargs='+', default=[0], help='random noise patterns (one process per sample)')
    n.add_argument('--control-samples', type=int, nargs='*', default=[0], help='samples that also run DPS every frame')
    e = sub.add_parser('enkf'); common(e)
    for q in (e, sub.add_parser('enkf_sweep')):
        if q is not e: common(q)
        q.add_argument('--enkf-members', type=int, default=20)
        q.add_argument('--enkf-obs-std', type=float, default=0.05, help='observation error R used by the EnKF (tuned separately from DPS obs_std)')
        q.add_argument('--init-level', type=float, default=0.1, help='initial ensemble spread, x reference std')
        q.add_argument('--demean-init', action='store_true', help='remove the mean of the initial perturbations (ensemble mean == centre)')
        q.add_argument('--center-file', default=None, help='npy cache of the shared frame-0 DPS analysis (loaded if present, else computed and saved)')
    e.add_argument('--schemes', nargs='+', default=['dps', 'enkf_plain', 'enkf_inflation', 'enkf_infl_loc'], help='dps must come first (it centres the EnKF)')
    e.add_argument('--inflation', type=float, default=1.2)
    e.add_argument('--radius-x', type=float, default=0.6)
    e.add_argument('--radius-z', type=float, default=0.3)
    w = sub.choices['enkf_sweep']
    w.add_argument('--inflations', type=float, nargs='+', default=[1.0, 1.2, 1.5, 2.0, 3.0])
    w.add_argument('--radii-x', type=float, nargs='+', default=[0.3, 0.6, 1.0], help='0 = no localization')
    w.add_argument('--radii-z', type=float, nargs='+', default=[0.2, 0.4, 0.8])
    sp = sub.add_parser('spinup'); common(sp)
    sp.add_argument('--inits', nargs='+', default=['zero', 'scale:0.3', 'later:60'])
    sp.add_argument('--enkf-members', type=int, default=20)
    sp.add_argument('--enkf-obs-std', type=float, default=0.1, help='sqrt of the diagonal of R')
    sp.add_argument('--init-level', type=float, default=1.0, help='initial ensemble spread, x reference std')
    sp.add_argument('--inflation', type=float, default=1.3)
    sp.add_argument('--radius-x', type=float, default=1.5)
    sp.add_argument('--radius-z', type=float, default=1.2)
    sp.add_argument('--obsonly-obs-std', type=float, default=0.1)
    sp.add_argument('--obsonly-rate', type=float, default=1.0)
    d = sub.add_parser('dps_sweep'); common(d)
    d.add_argument('--method', default='bg_obs', choices=['bg_obs', 'obs_only'])
    d.add_argument('--grid', nargs='+', required=True)
    r = sub.add_parser('recon'); common(r)
    r.add_argument('--frames', type=int, nargs='+', default=[10, 25, 40, 55, 70, 85])
    r.add_argument('--bg-errors', type=float, nargs='*', default=[], help='optional artificial smooth background errors (x reference std); off by default')
    r.add_argument('--model-rayleighs', type=float, nargs='+', default=[8e4, 5e4, 2e4], help='forecast-model Ra values (truth 1e5)')
    r.add_argument('--obsonly-obs-std', type=float, default=0.025, help='obs_std of the no-background arm (tuned on the validation case)')
    r.add_argument('--obsonly-rate', type=float, default=1.0)
    r.add_argument('--guidance-bg-std', type=float, default=0.3, help='bg_std of the background-as-guidance arm (bg_std=1 switches the guidance off in RB units)')
    r.add_argument('--guidance-obs-std', type=float, default=0.025)
    r.add_argument('--guidance-rate', type=float, default=1.0)
    r.add_argument('--model-leads', type=int, nargs='+', default=[1, 3, 6], help='frames of the Ra-mismatched model run')
    f = sub.add_parser('fusion'); common(f)
    f.add_argument('--obsonly-obs-std', type=float, default=0.025, help='obs_std of the observations-only arm (tuned on the validation case)')
    f.add_argument('--obsonly-rate', type=float, default=1.0)
    m = sub.add_parser('merge')
    m.add_argument('--inputs', nargs='+', required=True); m.add_argument('--output', required=True)
    a = p.parse_args()
    {'noise': noise, 'enkf': enkf, 'enkf_sweep': enkf_sweep, 'spinup': spinup, 'dps_sweep': dps_sweep, 'recon': recon, 'fusion': fusion, 'merge': merge}[a.cmd](a)


if __name__ == '__main__':
    main()
