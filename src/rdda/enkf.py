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


def enkf_analysis(ens, obs, mask, obs_std, *, inflation=1.0, loc_radius=None, periodic=(True, True), rng=None):
    """ens (N,H,W,C) forecast ensemble; obs (H,W,C) values; mask (H,W,C) bool; obs_std scalar or (C,).

    Returns the analysis ensemble (N,H,W,C). Observations are processed per observed pixel and variable
    (R diagonal), so the update is the exact sequential form of the formula above (observation-space
    perturbations kept consistent across members).
    """
    rng = np.random.default_rng(rng)
    n, h, w, c = ens.shape
    obs_std = np.broadcast_to(np.asarray(obs_std, dtype=np.float64), (c,))
    mean = ens.mean(0, keepdims=True)
    x = (mean + inflation * (ens - mean)).reshape(n, h * w, c).astype(np.float64)        # multiplicative inflation
    obs_pix = np.argwhere(mask.any(-1))
    rho_all = None if loc_radius is None else gaspari_cohn(_dist((h, w), obs_pix, periodic), loc_radius)
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
                   seed=0, periodic=(True, True), out=None):
    """EnKF rollout with the same bookkeeping/columns as rdda.cycle.run_cycle (forecast model = `forecaster`).

    Initial ensemble: (init_center if given, else truth[0]) + init_std * std_ref * N(0,1) on every variable (then made consistent).
    obs_std is relative to std_ref when relative_std. assim_every = K: observations assimilated at t=0 and
    every K steps (0 = only at t=0); in between the ensemble is just forecast.
    init_center: (H,W,C) first guess that already contains the t=0 observations -> the t=0 analysis is skipped
    (no double use of the data). The raw (pre-projection) balance of the analysis mean is stored as raw_*/raw_lp_*.
    """
    from rdda.assimilation import metric
    rng = np.random.default_rng(seed)
    std_ref = np.ones(truth.shape[-1]) if std_ref is None else np.asarray(std_ref, dtype=np.float64)
    ostd = np.asarray(obs_std) * (std_ref if relative_std else 1.0)
    spf = forecaster.steps_per_frame
    x0 = np.asarray(truth[0] if init_center is None else init_center, dtype=np.float32)
    ens = np.stack([forecaster.consistent(x0 + init_std * std_ref * rng.standard_normal(x0.shape).astype('f4'))
                    for _ in range(members)])
    rows = []
    for t in range(analysis_steps + 1):
        xtrue = np.asarray(truth[t * lead_frames], dtype=np.float32)
        if t > 0: ens = forecaster.forecast_many(ens, lead_frames * spf)
        bg = ens.mean(0)
        data_in = t == 0 or (assim_every > 0 and t % assim_every == 0)
        raw = {}
        if data_in and not (t == 0 and init_center is not None):
            ens = enkf_analysis(ens, np.where(mask, xtrue, 0), mask, ostd, inflation=inflation,
                                loc_radius=loc_radius, periodic=periodic, rng=rng)
            if hasattr(forecaster, 'balance'):
                m_ = ens.mean(0)
                raw = {**{f'raw_{k}': v for k, v in forecaster.balance(m_).items()},
                       **{f'raw_lp_{k}': v for k, v in forecaster.balance(m_, 16).items()}}
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
                                observations=np.where(mask, xtrue, 0), mask=mask, assimilated=np.array(data_in))
    return rows
