"""运行一个 RBC Case：CFL 自适应 dt、诊断时间序列、场快照、断点续算（含 p/tau；RK222 为单步法，续算与不中断一致）。
单进程：python run_case.py --case B --out DIR --t-end 100 [--nx --nz 覆盖] [--cfl 0.4] [--seed 0]
MPI（沿 z 切分，进程数<=nz，实测 256x128 在 32 进程附近最快）：OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader mpirun -n 32 python run_case.py ...；续算须用相同进程数。
初始扰动：b=(1-z)+amp*Σ 随机系数低波数模态（kx<=8，z 向 sin(mπz) m<=4），与分辨率无关；u=0（b 扰动不破坏不可压缩与边界条件）。
Nu=1+<u_z b>/kappa（ΔB=Lz=1）；Re=u_rms*Lz/nu；散度用 Dedalus 谱导数，相对量=rms(div u)/rms(|grad u|)。"""
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.fft import dct
import dedalus.public as d3
from mpi4py import MPI
from rdda.solver.rb_equations import RBConfig, build_solver
from cases import params
comm = MPI.COMM_WORLD; rank = comm.Get_rank()

def init_b(x, z, seed, amp=0.05, KX=8, MZ=4, lx=3.0):
    rng = np.random.default_rng(seed); b = 1 - z + 0 * x
    for k in range(1, KX + 1):
        for m in range(1, MZ + 1):
            a, ph = rng.standard_normal(), rng.uniform(0, 2 * np.pi)
            b = b + amp * a / np.sqrt(k * m) * np.sin(2 * np.pi * k * x / lx + ph) * np.sin(m * np.pi * z)
    return b

def cheb_coef(f):
    """(…,nz) 升序 Gauss 网格上的值 -> Chebyshev 系数 c_k（x=2z/Lz-1）。"""
    n = f.shape[-1]; y = dct(f[..., ::-1], type=2, axis=-1); c = y / n; c[..., 0] /= 2; return c

def zint(f, lz=1.0):
    c = cheb_coef(f); k = np.arange(c.shape[-1]); I = np.where(k % 2 == 0, 2 / (1 - np.where(k == 1, 2.0, k.astype(float)) ** 2), 0.0)
    return (c * I).sum(-1) * lz / 2

def wall_vals(f):
    c = cheb_coef(f); k = np.arange(c.shape[-1]); return (c * (-1.0) ** k).sum(-1), c.sum(-1)   # z=0, z=Lz

def wall_dz(f, lz=1.0):
    c = cheb_coef(f); k = np.arange(c.shape[-1]); return 2 / lz * (c * (-1.0) ** (k + 1) * k ** 2).sum(-1), 2 / lz * (c * k ** 2).sum(-1)

class Diag:
    def __init__(s, fields, cfg, nu, kappa):
        s.u, s.b, s.nu, s.kappa, s.cfg = fields['u'], fields['b'], nu, kappa, cfg
        s.gu = d3.grad(s.u)
        s.z = cfg.lz / 2 * (1 - np.cos(np.pi * (np.arange(cfg.nz) + .5) / cfg.nz)); s.dxg = cfg.lx / cfg.nx; s.dzg = np.gradient(s.z)
    def avg(s, f): return zint(f.mean(0), s.cfg.lz) / s.cfg.lz
    def __call__(s, t, dt):
        """所有 rank 必须同时调用（allgather 为集合通信）；只有 rank 0 返回诊断字典，其余返回 None。"""
        s.u.change_scales(1); s.b.change_scales(1)
        U = s.u.allgather_data('g'); b = s.b.allgather_data('g')
        gu = s.gu.evaluate(); gu.change_scales(1); G = gu.allgather_data('g')
        if rank != 0: return None
        ux, uz = U[0], U[1]; om = G[0, 1] - G[1, 0]; div = G[0, 0] + G[1, 1]
        ke = s.avg(0.5 * (ux ** 2 + uz ** 2)); gnorm2 = (G ** 2).sum((0, 1))
        bz0, bz1 = wall_dz(b); bw0, bw1 = wall_vals(b)
        return dict(t=t, dt=dt, ke=ke, re=np.sqrt(2 * ke) * s.cfg.lz / s.nu,
                    nu_conv=1 + s.avg(uz * b) / s.kappa, nu_bot=float(-bz0.mean()), nu_top=float(-bz1.mean()),
                    ens=s.avg(om ** 2), umax=float(max(abs(ux).max(), abs(uz).max())),
                    cfl=float(dt * (abs(ux) / s.dxg + abs(uz) / s.dzg).max()),
                    div_rel=float(np.sqrt(s.avg(div ** 2) / s.avg(gnorm2))),
                    uwall=float(max(abs(wall_vals(ux)[0]).max(), abs(wall_vals(ux)[1]).max(), abs(wall_vals(uz)[0]).max(), abs(wall_vals(uz)[1]).max())),
                    bwall=float(max(abs(bw0 - s.cfg.lz).max(), abs(bw1).max())), bprime2=s.avg((b - (s.cfg.lz - s.z)) ** 2))

def ckpt_path(out): return out / f'ckpt_r{rank:03d}.npz'

def save_ckpt(out, solver, dt, diag, n_diag, snaps_t):
    """每个 rank 存自己的系数数据；先全部写临时文件、同步后再原子替换，避免中断时各 rank 版本不一致。"""
    arr = {}
    for i, f in enumerate(solver.state): f.change_layout('c'); arr[f's{i}'] = f.data.copy()
    tmp = out / f'ckpt_r{rank:03d}.tmp.npz'
    np.savez(tmp, sim_time=solver.sim_time, iteration=solver.iteration, dt=dt, n_diag=n_diag, n_snap=len(snaps_t), snaps_t=json.dumps(snaps_t), nranks=comm.Get_size(),
             diag=json.dumps(diag if rank == 0 else []), **arr)
    comm.Barrier(); tmp.replace(ckpt_path(out)); comm.Barrier()

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--case', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--t-end', type=float, required=True); ap.add_argument('--nx', type=int); ap.add_argument('--nz', type=int)
    ap.add_argument('--cfl', type=float, default=0.4); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--diag-dt', type=float, default=0.1); ap.add_argument('--snap-dt', type=float, default=1.0)
    ap.add_argument('--ckpt-wall', type=float, default=300.); ap.add_argument('--max-dt', type=float, default=0.01)
    ap.add_argument('--ra', type=float, help='覆盖 Ra（与 --pr 一起，nu、kappa 按求解器定义重算）'); ap.add_argument('--pr', type=float)
    ap.add_argument('--exact-snaps', action='store_true', help='截断 dt 使快照时刻严格等于 k*snap-dt（帧间隔恒定）；旧结果未用此项')
    a = ap.parse_args(); P = params(a.case); nx, nz = a.nx or P['nx'], a.nz or P['nz']
    if a.ra: pr = a.pr or P['pr']; P = {**P, 'ra': a.ra, 'pr': pr, 'nu': (a.ra / pr) ** -0.5, 'kappa': (a.ra * pr) ** -0.5}
    out = Path(a.out)
    if rank == 0: (out / 'snap').mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    cfg = RBConfig(nx=nx, nz=nz, rayleigh=P['ra'], prandtl=P['pr'], dt=1e-4)
    solver, fields, grids = build_solver(cfg); x, z = grids          # x,z 为本 rank 的局部网格（可广播）
    diag_op = Diag(fields, cfg, P['nu'], P['kappa'])
    cfl = d3.CFL(solver, initial_dt=1e-4, cadence=5, safety=a.cfl, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=a.max_dt)
    cfl.add_velocity(fields['u'])
    diag, snaps_t, n_diag, dt = [], [], 0, 1e-4
    if ckpt_path(out).exists():
        z_ = np.load(ckpt_path(out)); assert int(z_['nranks']) == comm.Get_size(), f"续算须用相同进程数: ckpt {int(z_['nranks'])} vs {comm.Get_size()}"
        solver.sim_time = float(z_['sim_time']); solver.iteration = int(z_['iteration']); dt = float(z_['dt']); n_diag = int(z_['n_diag']); snaps_t = json.loads(str(z_['snaps_t']))
        for i, f in enumerate(solver.state): f.preset_layout('c'); f.data[:] = z_[f's{i}']
        cfl.stored_dt = dt   # 续算要恢复 CFL 的当前 dt，否则从 initial_dt 重新爬升，dt 序列与不中断的不同
        diag = json.loads(str(z_['diag']))
        if rank == 0: print(f'restart t={solver.sim_time:.3f}', flush=True)
    else:
        fields['b'].change_scales(1); fields['u'].change_scales(1)
        fields['b']['g'] = init_b(x, z, a.seed); fields['u']['g'] = 0
        for k in ('p', 'tau_p', 'tau_b1', 'tau_b2', 'tau_u1', 'tau_u2'): fields[k]['g'] = 0
        if rank == 0: (out / 'meta.json').write_text(json.dumps(dict(case=a.case, **{**P, "nx": nx, "nz": nz}, seed=a.seed, cfl_safety=a.cfl, max_dt=a.max_dt, dealias=cfg.dealias, nranks=comm.Get_size(),
            scheme='RK222', init='linear b + band-limited random b perturbation amp 0.05 (kx<=8, sin(m pi z) m<=4), u=0'), indent=1))
    t_d = n_diag * a.diag_dt; t_s = len(snaps_t) * a.snap_dt; wall = time.time(); last_ck = wall; n0 = solver.iteration
    while solver.sim_time < a.t_end:
        dt = cfl.compute_timestep()
        if a.exact_snaps and 1e-9 < t_s - solver.sim_time < dt * (1 + 1e-9): dt = t_s - solver.sim_time   # 恰好落在下一个快照时刻（CFL 内部的 stored_dt 不变）
        solver.step(dt)
        if solver.sim_time >= t_d:
            r = diag_op(solver.sim_time, dt)
            bad = comm.allreduce(int(not (np.isfinite(fields['b'].data).all() and np.isfinite(fields['u'].data).all())), op=MPI.SUM)
            if bad: raise FloatingPointError(f'nonfinite at t={solver.sim_time}')
            if rank == 0: diag.append(r)
            n_diag += 1; t_d += a.diag_dt
        if solver.sim_time >= t_s - (1e-9 if a.exact_snaps else 0.):
            fields['b'].change_scales(1); fields['u'].change_scales(1)
            B = fields['b'].allgather_data('g'); U = fields['u'].allgather_data('g')
            if rank == 0: np.save(out / 'snap' / f'{len(snaps_t):05d}.npy', np.stack([B, U[0], U[1]], -1).astype('f4'))
            snaps_t.append(solver.sim_time); t_s += a.snap_dt
        if solver.iteration % 50 == 0 and comm.bcast(time.time() - last_ck > a.ckpt_wall, root=0):
            save_ckpt(out, solver, dt, diag, n_diag, snaps_t); last_ck = time.time()
            if rank == 0: print(f't={solver.sim_time:.2f} it={solver.iteration} dt={dt:.2e} {(solver.iteration-n0)/(time.time()-wall):.1f} it/s Nu={diag[-1]["nu_conv"]:.2f} Re={diag[-1]["re"]:.0f}', flush=True)
    save_ckpt(out, solver, dt, diag, n_diag, snaps_t)
    if rank == 0:
        zf = diag_op.z; np.savez(out / 'diag.npz', **{k: np.array([r[k] for r in diag]) for k in diag[0]}, snaps_t=np.array(snaps_t), z=zf)
        print(f'done t={solver.sim_time:.2f} it={solver.iteration} wall={time.time()-wall:.0f}s ranks={comm.Get_size()}', flush=True)

if __name__ == '__main__': main()
