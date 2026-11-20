"""多进程预报器：一个成员由 R 个 MPI 进程合作积分（Dedalus 沿 z 分布），G 个成员组并行。
SPMD：mpirun -n (1+G*R)；rank 0 是主进程（EnKF/同化逻辑，接口同 RBForecaster），其余 rank 按 R 个一组分成 G 组，各组建一份分布式求解器，在 serve() 里等主进程派活。
用法（所有 rank 都执行同一脚本）：
    fc = MPIForecaster(config, procs_per_member=R, steps_per_frame=N)
    if not fc.is_master: fc.serve(); sys.exit(0)
    ...fc.forecast_many(ens, steps)...; fc.close()
与单进程 forecast(projection='none') 的差别只来自并行归约的舍入（见 analysis/rbc_explore/mpi_fc_test.py）。只支持不投影：输入应是（近似）合法的求解器状态。"""
import numpy as np
from mpi4py import MPI
from rdda.solver.rb_equations import build_solver


class MPIForecaster:
    def __init__(self, config, procs_per_member, steps_per_frame, world=None, adaptive=False, cfl_safety=0.4, max_dt=0.01):
        self.world = world or MPI.COMM_WORLD; rank, size = self.world.Get_rank(), self.world.Get_size()
        R = procs_per_member
        if size < 1 + R or (size - 1) % R: raise ValueError(f'需要 1+G*R 个进程（R={R}），现有 {size}')
        self.config, self.steps_per_frame, self.R, self.G = config, steps_per_frame, R, (size - 1) // R
        self.is_master = rank == 0; self._z = None
        self.group = self.world.Split(MPI.UNDEFINED if rank == 0 else (rank - 1) // R, rank)       # 主进程得到 COMM_NULL
        self.adaptive, self.cfl_safety, self.max_dt, self.last_nsteps = adaptive, cfl_safety, max_dt, []
        if not self.is_master:
            self.solver, self.fields, self.grids = build_solver(config, comm=self.group)
            import dedalus.public as d3          # CFL 自适应步长（与 run_case.py/生成数据相同的规则；进程组内全局归约）
            self.cfl = d3.CFL(self.solver, initial_dt=1e-4, cadence=5, safety=cfl_safety, threshold=0.05, max_change=1.5, min_change=0.5, max_dt=max_dt)
            self.cfl.add_velocity(self.fields['u'])

    # ---------- worker ----------
    def _run(self, state, steps, adaptive):
        c, f = self.config, self.fields
        init = np.asarray(state, dtype=np.float64).copy()
        zz = np.asarray(self._nodes()); low, high = int(np.argmin(zz)), int(np.argmax(zz))
        init[:, low, 0] = c.lz; init[:, high, 0] = 0.0                    # 与 forecast() 相同的 b 边界值
        f['b'].load_from_global_grid_data(init[..., 0])
        f['u'].load_from_global_grid_data(np.moveaxis(init[..., 1:3], -1, 0))
        for k in ('p', 'tau_p', 'tau_b1', 'tau_b2', 'tau_u1', 'tau_u2'): f[k]['g'] = 0
        if not adaptive:
            for _ in range(steps): self.solver.step(c.dt)
            n = steps
        else:                                  # 积分 span = steps*dt（steps 来自帧间隔/dt），步长由 CFL 决定；最后剩余时间平分，避免出现极小步
            target = self.solver.sim_time + steps * c.dt; self.cfl.stored_dt = 1e-4; n = 0
            while target - self.solver.sim_time > 1e-12:
                dt = self.cfl.compute_timestep(); rem = target - self.solver.sim_time
                dt = rem if dt >= rem else (rem / 2 if rem < 2 * dt else dt)
                self.solver.step(dt); n += 1
        bad = self.group.allreduce(int(not (np.isfinite(f['b'].data).all() and np.isfinite(f['u'].data).all())), op=MPI.SUM)
        if bad: return None, n
        f['b'].change_scales(1); f['u'].change_scales(1)
        B = f['b'].gather_data(root=0, layout='g'); U = f['u'].gather_data(root=0, layout='g')
        if self.group.Get_rank() != 0: return True, n
        return np.stack([B, U[0], U[1]], axis=-1).astype(np.float32), n

    def _nodes(self):
        from rdda.rb_noise import rb_nodes
        return rb_nodes(self.config.nz, self.config.lz)

    def serve(self):
        while True:
            msg = self.world.recv(source=0) if self.group.Get_rank() == 0 else None
            msg = self.group.bcast(msg, root=0)
            if msg[0] == 'stop': break
            _, items, steps, adaptive = msg; out = []
            for i, st in items: out.append((i,) + self._run(st, steps, adaptive))
            if self.group.Get_rank() == 0: self.world.send(out, dest=0)

    # ---------- master (same interface as RBForecaster) ----------
    def forecast_many(self, fields, steps, strict=True, adaptive=None):
        assert self.is_master
        jobs = [[] for _ in range(self.G)]
        for i, f in enumerate(fields): jobs[i % self.G].append((i, np.asarray(f, dtype=np.float32)))
        act = [g for g in range(self.G) if jobs[g]]
        ad = self.adaptive if adaptive is None else adaptive
        for g in act: self.world.send(('run', jobs[g], steps, ad), dest=1 + g * self.R)
        res = [None] * len(fields); self.last_nsteps = [0] * len(fields)
        for g in act:
            for i, r, n in self.world.recv(source=1 + g * self.R):
                self.last_nsteps[i] = n
                if r is None:
                    if strict: raise FloatingPointError('Non-finite Dedalus step after restart')
                    r = np.full((self.config.nx, self.config.nz, 3), np.nan, np.float32)      # strict=False: 发散的成员记为 NaN
                res[i] = r
        return np.stack(res)

    def __call__(self, analysis, steps): return self.forecast_many([analysis], steps)[0], {}

    def consistent(self, field): return field

    def balance(self, field):
        from rdda.diagnostics import rb_balance
        return rb_balance(field, self._nodes(), self.config.lx, self.config.lz)

    def close(self):
        if self.is_master:
            for g in range(self.G): self.world.send(('stop', None, None, None), dest=1 + g * self.R)
