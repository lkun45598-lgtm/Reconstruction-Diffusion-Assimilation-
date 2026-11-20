"""批量运行 rbc_dps_run.py：任务队列 + 核段槽位（每槽 1+R 个核）+ 按空闲显存动态选 GPU；已完成的任务（state.npz 的 t == frames）自动跳过，中断后重跑即续跑。
python rbc_dps_queue.py --jobs jobs.json --slots 0-4,5-9,... [--gpus 0,1,...] [--min-free 9000]
jobs.json: [{"name":..., "truth":DIR, "donor":DIR, "out":DIR, "prior":PT, "K":10, "frames":200, "bg_std":1.0}, ...]（K=0 = 自由预报，不占 GPU）"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--jobs', required=True); ap.add_argument('--slots', required=True); ap.add_argument('--gpus', default='')
ap.add_argument('--R', type=int, default=4); ap.add_argument('--min-free', type=int, default=9000, help='启动一个 DPS 任务所需的 GPU 空闲显存（MiB）'); ap.add_argument('--reserve', type=int, default=800, help='每个已分配到该 GPU 的任务预扣的显存（MiB；DPS 用文件锁串行，平时只占上下文+权重）'); ap.add_argument('--conc', type=int, default=1, help='每张 GPU 同时做 DPS 的进程数'); ap.add_argument('--python', default=sys.executable); ap.add_argument('--extra', default='', help='附加给 rbc_dps_run.py 的参数')
a = ap.parse_args()
jobs = json.load(open(a.jobs)); slots = a.slots.split(','); allowed = a.gpus.split(',') if a.gpus else None; root = Path(__file__).resolve().parent.parent
env = dict(os.environ, OMP_NUM_THREADS='1', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader', PYTHONPATH=str(root / 'src'), PYTHONUNBUFFERED='1')
def finished(j):
    try: return int(np.load(Path(j['out']) / 'state.npz')['t']) >= j['frames']
    except Exception: return False
def pick_gpu(taken):
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split('\n')
    best = None
    for l in q:
        if not l.strip(): continue
        i, tot, used = [x.strip() for x in l.split(',')]; free = int(tot) - int(used) - a.reserve * taken.get(i, 0)
        if (allowed is None or i in allowed) and free >= a.min_free and (best is None or (taken.get(i, 0), -free) < best[1]): best = (i, (taken.get(i, 0), -free))   # 已分到任务最少的卡优先（DPS 在每张卡上串行，要均摊），同数时空闲显存多的优先
    return best[0] if best else None
pending = [j for j in jobs if not finished(j)]; run = {}; gpu_of = {}
print(f'{len(pending)}/{len(jobs)} jobs pending, {len(slots)} slots', flush=True)
while pending or run:
    for s in list(run):
        p, j = run[s]
        if p.poll() is not None: print(time.strftime('%H:%M'), 'END', j['name'], 'rc', p.returncode, 'finished' if finished(j) else 'NOT FINISHED', flush=True); gpu_of.pop(s, None); del run[s]
    for s in slots:
        if s not in run and pending:
            j = pending[0]; g = None
            if j['K'] > 0:
                taken = {}
                for v in gpu_of.values(): taken[v] = taken.get(v, 0) + 1
                g = pick_gpu(taken)
                if g is None: continue
            pending.pop(0); lo, hi = map(int, s.split('-')); ranks = hi - lo + 1; ch = int(j.get('chains', 1)); assert ranks == 1 + a.R * ch, f'槽位 {s} 需要 1+R*chains={1 + a.R * ch} 个核'
            Path(j['out']).mkdir(parents=True, exist_ok=True)
            cmd = ['taskset', '-c', s, str(Path(a.python).parent / 'mpirun'), '--bind-to', 'none', '-n', str(ranks), a.python, str(root / 'analysis/rbc_dps_run.py'), '--truth', j['truth'], '--donor', j['donor'], '--out', j['out'], '--prior', j['prior'],
                   '--frames', str(j['frames']), '--K', str(j['K']), '--bg-std', str(j['bg_std']), '--R', str(a.R), '--device', 'cuda:0'] + (['--chains', str(ch)] if ch > 1 else []) + a.extra.split() + [str(x) for x in j.get('extra', [])]
            e = dict(env, RBC_DPS_CONC=str(a.conc));
            if g is not None: e['CUDA_VISIBLE_DEVICES'] = g; gpu_of[s] = g
            run[s] = (subprocess.Popen(cmd, stdout=open(Path(j['out']) / 'log.txt', 'a'), stderr=subprocess.STDOUT, env=e, cwd=root), j); print(time.strftime('%H:%M'), 'START', j['name'], 'slot', s, 'gpu', g, flush=True); time.sleep(8)
    time.sleep(15)
print('ALL FINISHED', flush=True)
