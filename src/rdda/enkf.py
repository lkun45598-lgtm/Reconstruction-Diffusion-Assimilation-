"""Stochastic ensemble Kalman filter for sparse point observations, with inflation and Gaspari-Cohn localization.

Analysis (the formula of the slide):   x^a - x^f = P^f H^T (H P^f H^T + R)^-1 [ y - H(x^f) ] = X' c
P^f is the sample covariance of the forecast ensemble (anomalies X'), H picks the observed pixels. Each member
assimilates its own perturbed observation y + eps, eps ~ N(0, R) (stochastic EnKF). Localization multiplies the
covariance entries by a compactly supported Gaspari-Cohn function of the distance to the observation, so one
observation only updates its neighbourhood. State = the model variables stacked on the last axis (H,W,C); the
covariance between variables is kept (cross-variable updates), distances are periodic in `periodic` axes.
"""
import numpy as np


def gaspari_cohn(dist, radius):
    """Gaspari & Cohn (1999) 5th-order compactly supported correlation, support = 2*radius."""
    z = np.abs(dist) / radius
    out = np.zeros_like(z, dtype=np.float64)
    a, b = z <= 1, (z > 1) & (z < 2)
    za, zb = z[a], z[b]
    out[a] = -0.25 * za ** 5 + 0.5 * za ** 4 + 0.625 * za ** 3 - 5 / 3 * za ** 2 + 1
    out[b] = 1 / 12 * zb ** 5 - 0.5 * zb ** 4 + 0.625 * zb ** 3 + 5 / 3 * zb ** 2 - 5 * zb + 4 - 2 / 3 / zb
    return out


def _dist(shape, obs_ij, periodic):
    """Grid distance (in pixels) from every grid point to every observed pixel -> (n_obs, H*W)."""
    h, w = shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    di = np.abs(ii.ravel()[None] - obs_ij[:, :1])
    dj = np.abs(jj.ravel()[None] - obs_ij[:, 1:])
    if periodic[0]: di = np.minimum(di, h - di)
    if periodic[1]: dj = np.minimum(dj, w - dj)
    return np.sqrt(di.astype(np.float64) ** 2 + dj.astype(np.float64) ** 2)


def _dist_phys(coords, obs_ij, lengths, radii):
    """Anisotropic normalised distance sqrt((dx/rx)^2 + (dz/rz)^2) from every grid point to every observed pixel on a
    grid with PHYSICAL coordinates (e.g. RB: periodic uniform x, non-uniform Chebyshev z). lengths[a] = period or None."""
    out = 0.
    for a, (c, length, r) in enumerate(zip(coords, lengths, radii)):
        c = np.asarray(c, dtype=np.float64)
        d = np.abs(c[None, :] - c[obs_ij[:, a]][:, None])                      # (n_obs, n_axis)
        if length: d = np.minimum(d, length - d)
        out = out + ((d / r) ** 2)[:, :, None] if a == 0 else out + ((d / r) ** 2)[:, None, :]
    return np.sqrt(out).reshape(len(obs_ij), -1)


def enkf_analysis(ens, obs, mask, obs_std, *, inflation=1.0, loc_radius=None, periodic=(True, True), rng=None,
                  coords=None, lengths=(None, None)):
    """ens (N,H,W,C) forecast ensemble; obs (H,W,C) values; mask (H,W,C) bool; obs_std scalar or (C,).

    Returns the analysis ensemble (N,H,W,C). Observations are processed per observed pixel and variable
    (R diagonal), so the update is the exact sequential form of the formula above (observation-space
    perturbations kept consistent across members).
    Localization: loc_radius scalar in PIXELS (uniform grids, `periodic` flags), or -- when `coords` (physical
    coordinates of axis 0 and 1) is given -- loc_radius = (r0, r1) in physical units with `lengths` = periods.
    """
    rng = np.random.default_rng(rng)
    n, h, w, c = ens.shape
    obs_std = np.broadcast_to(np.asarray(obs_std, dtype=np.float64), (c,))
    mean = ens.mean(0, keepdims=True)
    x = (mean + inflation * (ens - mean)).reshape(n, h * w, c).astype(np.float64)        # multiplicative inflation
    obs_pix = np.argwhere(mask.any(-1))
    if loc_radius is None: rho_all = None
    elif coords is not None: rho_all = gaspari_cohn(_dist_phys(coords, obs_pix, lengths, np.broadcast_to(loc_radius, (2,))), 1.0)
    else: rho_all = gaspari_cohn(_dist((h, w), obs_pix, periodic), loc_radius)
    for k, (i, j) in enumerate(obs_pix):
        p = i * w + j
        for ch in range(c):
            if not mask[i, j, ch]: continue
            hx = x[:, p, ch]                                         # H x_i  (N,)
            a = x - x.mean(0, keepdims=True)
            ha = hx - hx.mean()
            var = (ha @ ha) / (n - 1) + obs_std[ch] ** 2             # H P H^T + R  (scalar)
            cov = np.einsum('n,npc->pc', ha, a) / (n - 1)            # P H^T  (all variables, all pixels)
            if rho_all is not None: cov = cov * rho_all[k][:, None]
            y = obs[i, j, ch] + obs_std[ch] * rng.standard_normal(n)  # perturbed observations
            x += (cov[None] / var) * (y - hx)[:, None, None]
    return x.reshape(n, h, w, c).astype(np.float32)


def run_enkf_cycle(truth, forecaster, mask, *, members=20, obs_std=0.05, relative_std=True, std_ref=None,
                   init_std=0.0, init_center=None, inflation=1.1, loc_radius=None, analysis_steps=20, lead_frames=1, assim_every=1,
                   seed=0, periodic=(True, True), out=None, coords=None, lengths=(None, None), perturb=None,
                   hard_observation=False, lowpass_k=16, obs_noise=0.0, obs_seed=0, skip_t0=True, demean_init=False, init_ensemble=None):
    """EnKF rollout with the same bookkeeping/columns as rdda.cycle.run_cycle (forecast model = `forecaster`).

    Initial ensemble: (init_center if given, else truth[0]) + init_std * std_ref * N(0,1) on every variable (then made consistent).
    obs_std is relative to std_ref when relative_std. assim_every = K: observations assimilated at t=0 and
    every K steps (0 = only at t=0); in between the ensemble is just forecast.
    init_center: (H,W,C) first guess that already contains the t=0 observations -> the t=0 analysis is skipped
    (no double use of the data). The raw (pre-projection) balance of the analysis mean is stored as raw_*/raw_lp_*
    (raw_lp_* only if lowpass_k is not None; the low-pass needs spectral periodic axes, so RB passes None).
    perturb(rng) -> (H,W,C): physically consistent initial perturbation (default: white noise, std = init_std*std_ref).
    hard_observation: after each analysis shift every member at observed entries by (obs - ensemble mean), i.e. the
    mean is overwritten by the observation like DPS's hard_observation, spread is kept.
    obs_noise: std of the Gaussian error added to the observations (same draws as rdda.cycle.run_cycle; every member's
    perturbed copy adds obs_std on top, i.e. R = diag(obs_std^2)). skip_t0=False: also analyse t=0 with init_center as first guess.
    coords/lengths: physical-coordinate localization (see enkf_analysis).
    """
    from rdda.assimilation import metric, noisy_observations
    rng = np.random.default_rng(seed)
    std_ref = np.ones(truth.shape[-1]) if std_ref is None else np.asarray(std_ref, dtype=np.float64)
    ostd = np.asarray(obs_std) * (std_ref if relative_std else 1.0)
    spf = forecaster.steps_per_frame
    x0 = np.asarray(truth[0] if init_center is None else init_center, dtype=np.float32)
    pert = perturb if perturb is not None else (lambda r: init_std * std_ref * r.standard_normal(x0.shape))
    if init_ensemble is not None:                                          # given states (e.g. snapshots of other times), no centre + perturbation
        assert len(init_ensemble) == members, 'init_ensemble must have `members` states'
        ens = np.stack([forecaster.consistent(np.asarray(e, dtype=np.float32)) for e in init_ensemble])
    else:
        perts = np.stack([np.asarray(pert(rng), dtype=np.float32) for _ in range(members)])
        if demean_init: perts = perts - perts.mean(0, keepdims=True)       # ensemble mean == init_center exactly
        ens = np.stack([forecaster.consistent(x0 + p_) for p_ in perts])
    rows = []
    for t in range(analysis_steps + 1):
        xtrue = np.asarray(truth[t * lead_frames], dtype=np.float32)
        if t > 0: ens = forecaster.forecast_many(ens, lead_frames * spf)
        bg = ens.mean(0)
        data_in = t == 0 or (assim_every > 0 and t % assim_every == 0)
        raw = {}
        obs = noisy_observations(xtrue, mask, obs_noise, obs_seed, t)
        if data_in and not (t == 0 and init_center is not None and skip_t0):
            ens = enkf_analysis(ens, obs, mask, ostd, inflation=inflation,
                                loc_radius=loc_radius, periodic=periodic, rng=rng, coords=coords, lengths=lengths)
            if hard_observation:
                ens[:, mask] += (obs - ens.mean(0))[mask]
            if hasattr(forecaster, 'balance'):
                m_ = ens.mean(0)
                raw = {f'raw_{k}': v for k, v in forecaster.balance(m_).items()}
                if lowpass_k is not None: raw.update({f'raw_lp_{k}': v for k, v in forecaster.balance(m_, lowpass_k).items()})
            ens = np.stack([forecaster.consistent(e) for e in ens])
        an = ens.mean(0)
        miss = ~mask
        rows.append({'frame': t, 'truth_frame': t * lead_frames, 'lead_frames': lead_frames, 'assimilated': int(data_in),
                     'analysis_rmse': metric(an, xtrue), 'missing_rmse': metric(an, xtrue, miss),
                     'observed_rmse': metric(an, xtrue, mask),
                     **{f'missing_rmse_ch{k}': metric(an[..., k:k + 1], xtrue[..., k:k + 1], miss[..., k:k + 1])
                        for k in range(truth.shape[-1])},
                     'background_rmse': None if t == 0 else metric(bg, xtrue),
                     **{f'bg_missing_rmse_ch{k}': None if t == 0 else
                        metric(bg[..., k:k + 1], xtrue[..., k:k + 1], miss[..., k:k + 1]) for k in range(truth.shape[-1])},
                     'ensemble_spread': float(np.sqrt(np.mean(ens.var(0, ddof=1)))), 'projection_rmse': None, **raw})
        if out is not None:
            np.savez_compressed(f'{out}/frame_{t:04d}.npz', truth=xtrue,
                                background=np.zeros((0,), 'f4') if t == 0 else bg, analysis=an,
                                observations=obs, mask=mask, assimilated=np.array(data_in))
    return rows
