"""Denoising path of one DPS assimilation (new setting: donor chain, first assimilation at frame 20, then every K frames).
Runs case CASE up to the assimilation at FRAME and records x_t and x0_hat at every sampler step (member 0 and the member mean; physical units),
then writes into OUT/: inputs_result.png (truth | background | sparse observations | analysis), steps_x0.png, steps_xt.png (selected steps, rows b/ux/uz),
steps/ (one png per step, 3 variables), denoise_x0.gif, denoise.npz.
python analysis/dps_denoise_steps.py DEVICE CASE FRAME OUT [--k 10 --bg-std 0.4 --seed 42 --ncols 8]"""
import argparse, glob, io, sys
import numpy as np
sys.path.insert(0, 'analysis'); import study_rb as S
import rdda.cycle as C
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from rdda.rb_noise import rb_nodes
p = argparse.ArgumentParser(); p.add_argument('device'); p.add_argument('case', type=int); p.add_argument('frame', type=int); p.add_argument('out'); p.add_argument('--k', type=int, default=10); p.add_argument('--bg-std', type=float, default=0.4)
p.add_argument('--seed', type=int, default=42); p.add_argument('--ncols', type=int, default=8); A = p.parse_args(); out = Path(A.out); (out / 'steps').mkdir(parents=True, exist_ok=True); FIRST = 20
a = argparse.Namespace(data='data/processed_600', split='test', case_index=A.case, prior='outputs/rb_prior_4gpu/best.pt', out='x', start_frame=0, analysis_steps=A.frame, rayleigh=1e5, model_rayleigh=5e4,
    observation_fraction=.01, members=32, dps_steps=100, obs_std=.05, bg_std=A.bg_std, dps_rate=2., init='noise', t_start=.4, hard_observation=False, div_weight=.3,
    obs_noise=.1, every=A.k, seed=A.seed, device=A.device, enkf_members=20, enkf_obs_std=0.1, init_level=0.1, demean_init=True)
truth, mask, fc = S.setup(a)
train = sorted(glob.glob('data/processed/train/case_*.npy')); rng = np.random.default_rng(A.case); donor = int(rng.integers(len(train))); fr = int(rng.integers(0, 80))
init = fc.consistent(np.asarray(np.load(train[donor], mmap_mode='r')[fr], dtype=np.float32))
calls = []; real = C.assimilate
def spy(prior, obs, m, bg, **kw):   # record the path of every assimilation, keep the one at A.frame (the last call)
    rec = []; r = real(prior, obs, m, bg, trace=lambda t, xt, x0: rec.append((t, xt[0], xt[:].mean(0), x0[0], x0.mean(0))), **kw); calls.append(dict(rec=rec, obs=obs, bg=bg, result=r)); return r
C.assimilate = spy
out_rows = C.run_cycle(truth, a.prior, fc, mask, analysis_steps=A.frame, method='bg_obs', assim_every=A.k, first_assim=FIRST, init_state=init,
                       **{**S.cyc_kw(a), **S.dps_kw(a, project_output=True)})
assert out_rows[-1]['assimilated'] == 1, 'FRAME is not an assimilation frame'
c = calls[-1]; rec = c['rec']; ts = np.array([r[0] for r in rec]); xt0, xtm, x00, x0m = (np.stack([r[i] for r in rec]) for i in (1, 2, 3, 4))   # (steps,3,H,W)
tru = np.asarray(truth[A.frame], np.float32); obs = c['obs']; bg = c['bg']; ana = c['result']['mean']
np.savez_compressed(out / 'denoise.npz', t=ts, xt_member0=xt0, xt_mean=xtm, x0hat_member0=x00, x0hat_mean=x0m, truth=tru, background=bg, observations=obs, mask=mask, analysis=ana, frame=A.frame, K=A.k, bg_std=A.bg_std)
nx, nz = tru.shape[:2]; xe = np.linspace(0, 3., nx + 1); z = rb_nodes(nz); ze = np.concatenate([[0], (z[1:] + z[:-1]) / 2, [1.]]); NAMES = ['b', '$u_x$', '$u_z$']
lim = [(0, 1)] + [(-float(np.abs(tru[..., r]).max()) * 1.1,) * 1 + (float(np.abs(tru[..., r]).max()) * 1.1,) for r in (1, 2)]; lim[1] = (-lim[1][1], lim[1][1]); lim[2] = (-lim[2][1], lim[2][1])
def show(ax, f, r, **kw): return ax.pcolormesh(xe, ze, np.asarray(f).T, cmap='RdBu_r', vmin=lim[r][0], vmax=lim[r][1], shading='flat', **kw)
# 1) inputs and result
fig, ax = plt.subplots(3, 4, figsize=(13, 5.6), constrained_layout=True)
for r in range(3):
    for j, (ttl, f) in enumerate((('truth', tru), ('background (solver forecast)', bg), ('sparse observations', None), ('DPS analysis', ana))):
        if f is None:
            ax[r, j].set_facecolor('#f2f2f2'); xs, zs = np.nonzero(mask[..., r]); xc = (xe[:-1] + xe[1:]) / 2; ax[r, j].scatter(xc[xs], z[zs], c=obs[xs, zs, r], cmap='RdBu_r', vmin=lim[r][0], vmax=lim[r][1], s=40, edgecolor='k', lw=.5); ax[r, j].set_xlim(0, 3); ax[r, j].set_ylim(0, 1)
        else: m = show(ax[r, j], f[..., r], r)
        if r == 0: ax[r, j].set_title(ttl, fontsize=9)
        if j == 0: ax[r, j].set_ylabel(NAMES[r]);
        ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
fig.suptitle(f'case {A.case}, frame {A.frame} (K={A.k}, bg_std={A.bg_std})', fontsize=10); fig.savefig(out / 'inputs_result.png', dpi=120); plt.close(fig)
# 2) selected steps
sel = np.unique(np.linspace(0, len(ts) - 1, A.ncols).round().astype(int))
for name, arr in (('x0', x00), ('xt', xt0)):
    fig, ax = plt.subplots(3, len(sel), figsize=(1.9 * len(sel), 3.3), constrained_layout=True)
    for j, k in enumerate(sel):
        for r in range(3):
            ax[r, j].pcolormesh(xe, ze, arr[k, r].T, cmap='RdBu_r', vmin=lim[r][0] if name == 'x0' else None, vmax=lim[r][1] if name == 'x0' else None, shading='flat'); ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[0, j].set_title(f't={ts[k]:.2f}', fontsize=8)
    for r in range(3): ax[r, 0].set_ylabel(NAMES[r])
    fig.suptitle('x0_hat (member 0)' if name == 'x0' else 'x_t (member 0, own colour scale per panel)', fontsize=9); fig.savefig(out / f'steps_{name}.png', dpi=130); plt.close(fig)
# 3) per-step pngs and gif
imgs = []
for k in range(len(ts)):
    fig, ax = plt.subplots(3, 2, figsize=(7.4, 4.6), constrained_layout=True)
    for r in range(3):
        show(ax[r, 0], x00[k, r], r); ax[r, 1].pcolormesh(xe, ze, xt0[k, r].T, cmap='RdBu_r', shading='flat'); ax[r, 0].set_ylabel(NAMES[r]); ax[r, 0].set_xticks([]); ax[r, 0].set_yticks([]); ax[r, 1].set_xticks([]); ax[r, 1].set_yticks([])
    ax[0, 0].set_title('x0_hat', fontsize=9); ax[0, 1].set_title('x_t', fontsize=9); fig.suptitle(f'step {k + 1}/{len(ts)}  t={ts[k]:.3f}', fontsize=10)
    fig.savefig(out / 'steps' / f'step_{k:03d}.png', dpi=100); b = io.BytesIO(); fig.savefig(b, format='png', dpi=100); plt.close(fig); b.seek(0); imgs.append(Image.open(b).convert('P', palette=Image.ADAPTIVE))
imgs[0].save(out / 'denoise_x0.gif', save_all=True, append_images=imgs[1:] + [imgs[-1]] * 8, duration=120, loop=0)
print('saved', out, len(ts), 'steps; rmse(missing) of analysis', out_rows[-1]['missing_rmse'], 'background', out_rows[-1]['background_rmse'])
