"""批量生成 RB 数据集（Pr 固定，只变 Ra）：任务列表 + 本机槽位调度（每槽 16 进程 MPI，绑核，可断点续算）。
每条轨迹输出 ROOT/ra<Ra>/<split>_s<seed>/{traj.npy (T,nx,nz,3) float32, times.npy, diag.npz, meta.json}；完成后校验 traj 与逐帧快照一致再删除 snap/。
python gen_dataset.py --root DIR --slots 0-15,16-31,... [--only-first N] [--list]"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
PR = 0.3; NX, NZ = 512, 256; T_END = 260.0; SNAP = 0.35
TRAIN_RA = [3e6, 5.5e6, 1e7, 1.8e7, 3e7, 5.5e7, 1e8]; N_TRAIN = 23
VAL_RA = [3e6, 1e7, 3e7, 1e8]; N_VAL = 5
TEST_RA_IN = [3e6, 1e7, 3e7, 1e8]; N_TEST_IN = 20
TEST_RA_OFF = [4e6, 1.4e7, 4e7]; N_TEST_OFF = 10
def jobs():
    J = []
    for i, ra in enumerate(TRAIN_RA): J += [dict(ra=ra, split='train', seed=100000 + 1000 * i + k) for k in range(N_TRAIN)]
    for i, ra in enumerate(VAL_RA): J += [dict(ra=ra, split='val', seed=200000 + 1000 * i + k) for k in range(N_VAL)]
    for i, ra in enumerate(TEST_RA_IN): J += [dict(ra=ra, split='test', seed=300000 + 1000 * i + k) for k in range(N_TEST_IN)]
    for i, ra in enumerate(TEST_RA_OFF): J += [dict(ra=ra, split='testoff', seed=400000 + 1000 * i + k) for k in range(N_TEST_OFF)]
    # 每个 (Ra,split) 先出 1 条再出其余，最贵的 Ra 先排，便于尽早检查并均衡负载
    first = {}; order = []
    for j in J: first.setdefault((j['ra'], j['split']), len(order)); order.append(j)
    pri = lambda j: (0 if order.index(j) == first[(j['ra'], j['split'])] else 1, -j['ra'])
    return sorted(J, key=pri)
# 相对成本（每个时间单位的耗时，以 Ra=1e8 为 1；3e7 为实测 0.74，其余是估计，只用于负载均衡）
COST = {1e8: 1., 5.5e7: .85, 4e7: .78, 3e7: .74, 1.8e7: .65, 1.4e7: .6, 1e7: .5, 5.5e6: .45, 4e6: .42, 3e6: .4}
def shard(J, weights, force_first_to=None, n_force=0):
    """确定性分配：按优先级顺序，每条放到 (已分配成本+本条成本)/权重 最小的机器；前 n_force 条强制给 force_first_to（本机已在跑的）。"""
    load = {m: 0. for m in weights}; out = {m: [] for m in weights}
    for i, j in enumerate(J):
        m = force_first_to if i < n_force else min(weights, key=lambda k: (load[k] + COST[j['ra']]) / weights[k])
        load[m] += COST[j['ra']]; out[m].append(j)
    return out, load
def jdir(root, j): return Path(root) / f"ra{j['ra']:.3g}" / f"{j['split']}_s{j['seed']}"
def finalize(d):
    snaps = sorted((d / 'snap').glob('*.npy')); T = np.stack([np.load(f) for f in snaps]); np.save(d / 'traj.tmp.npy', T)
    chk = np.load(d / 'traj.tmp.npy', mmap_mode='r')
    assert chk.shape == T.shape and all(np.array_equal(chk[i], np.load(snaps[i])) for i in (0, len(snaps) // 2, len(snaps) - 1))
    (d / 'traj.tmp.npy').replace(d / 'traj.npy'); np.save(d / 'times.npy', np.load(d / 'diag.npz')['snaps_t'])
    for f in snaps: f.unlink()
    (d / 'snap').rmdir(); (d / 'DONE').write_text(f'{T.shape}\n')
def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--root', required=True); ap.add_argument('--slots', required=True)
    ap.add_argument('--list', action='store_true'); ap.add_argument('--only-first', type=int, default=0); ap.add_argument('--python', default=sys.executable); ap.add_argument('--machines', help='name:weight,... 例如 local:192,h114:256,h33:144'); ap.add_argument('--me'); ap.add_argument('--force-local-first', type=int, default=0); ap.add_argument('--assign', help='JSON {机器: [raXXX/split_sSEED,...]}，按给定顺序执行；优先于 --machines'); ap.add_argument('--t-end', type=float, default=T_END)
    a = ap.parse_args(); J = jobs()
    if a.list:
        import collections; c = collections.Counter((j['split'], j['ra']) for j in J); [print(k, v) for k, v in sorted(c.items())]; print('total', len(J)); return
    if a.only_first: J = J[:a.only_first]
    if a.assign:
        A = json.loads(Path(a.assign).read_text())[a.me]; byk = {f"ra{j['ra']:.3g}/{j['split']}_s{j['seed']}": j for j in J}; J = [byk[k] for k in A]
        print('assign:', len(J), 'jobs for', a.me, flush=True)
    if a.machines:
        W = {k: float(v) for k, v in (x.split(':') for x in a.machines.split(','))}; sh, load = shard(J, W, 'local', a.force_local_first)
        print('shard (jobs, relative cost, cost/weight):', {m: (len(sh[m]), round(load[m], 1), round(load[m] / W[m], 3)) for m in W}, flush=True); J = sh[a.me]
    slots = a.slots.split(','); run = {}; pending = [j for j in J if not (jdir(a.root, j) / 'DONE').exists()]; failed = []
    env = dict(os.environ, OMP_NUM_THREADS='1', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader')
    print(f'{len(pending)} jobs pending, {len(slots)} slots', flush=True)
    while pending or run:
        for s in list(run):
            p, j = run[s]
            if p.poll() is not None:
                d = jdir(a.root, j)
                if p.returncode == 0 and (d / 'diag.npz').exists(): finalize(d); print(time.strftime('%H:%M'), 'DONE', d.name, d.parent.name, flush=True)
                else: failed.append(j); print(time.strftime('%H:%M'), 'FAILED', d, 'rc', p.returncode, flush=True)
                del run[s]
        for s in slots:
            if s not in run and pending:
                j = pending.pop(0); d = jdir(a.root, j); d.mkdir(parents=True, exist_ok=True); lo, hi = s.split('-'); n = int(hi) - int(lo) + 1
                cmd = ['taskset', '-c', s, str(Path(a.python).parent / 'mpirun'), '--bind-to', 'none', '-n', str(n), a.python, 'run_case.py', '--case', 'S4', '--ra', str(j['ra']), '--pr', str(PR), '--nx', str(NX), '--nz', str(NZ),
                       '--seed', str(j['seed']), '--t-end', str(a.t_end), '--snap-dt', str(SNAP), '--exact-snaps', '--out', str(d), '--ckpt-wall', '600']
                run[s] = (subprocess.Popen(cmd, stdout=open(d / 'log.txt', 'a'), stderr=subprocess.STDOUT, env=env, cwd=Path(__file__).parent), j); print(time.strftime('%H:%M'), 'START', d.parent.name, d.name, 'slot', s, flush=True)
        time.sleep(15)
    print('ALL FINISHED; failed:', len(failed), flush=True); [print(' ', jdir(a.root, j)) for j in failed]
if __name__ == '__main__': main()
