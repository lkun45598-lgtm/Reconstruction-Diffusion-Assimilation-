"""批量运行 rbc_enkf_run.py：任务队列 + 核段槽位 + GPU 轮转；已完成的任务（state.npz 的循环数 == frames）自动跳过，中断后重跑本脚本即续跑。
python rbc_enkf_queue.py --jobs jobs.json --slots 0-40,41-81,... --gpus 4,5,6 [--R 2]
jobs.json: [{"name":..., "truth":DIR, "donor":DIR, "out":DIR, "args":["--K","10","--inflation","1.1",...]}, ...]"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--jobs', required=True); ap.add_argument('--slots', required=True); ap.add_argument('--gpus', default='')
ap.add_argument('--R', type=int, default=2); ap.add_argument('--members', type=int, default=20); ap.add_argument('--python', default=sys.executable); a = ap.parse_args()
jobs = json.load(open(a.jobs)); slots = a.slots.split(','); gpus = a.gpus.split(',') if a.gpus else [None]; root = Path(__file__).resolve().parent.parent
env = dict(os.environ, OMP_NUM_THREADS='1', OMPI_MCA_pml='ob1', OMPI_MCA_btl='self,vader', PYTHONPATH=str(root / 'src'), PYTHONUNBUFFERED='1')
def finished(j):
    s = Path(j['out']) / 'state.npz'; fr = int(j['args'][j['args'].index('--frames') + 1])
    try: return int(np.load(s)['t']) >= fr
    except Exception: return False
pending = [j for j in jobs if not finished(j)]; run = {}; k = 0
print(f'{len(pending)}/{len(jobs)} jobs pending, {len(slots)} slots', flush=True)
while pending or run:
    for s in list(run):
        p, j = run[s]
        if p.poll() is not None: print(time.strftime('%H:%M'), 'END', j['name'], 'rc', p.returncode, 'finished' if finished(j) else 'NOT FINISHED', flush=True); del run[s]
    for s in slots:
        if s not in run and pending:
            j = pending.pop(0); lo, hi = map(int, s.split('-')); ranks = hi - lo + 1; g = gpus[k % len(gpus)]; k += 1
            assert (ranks - 1) % a.R == 0 and (ranks - 1) // a.R == a.members, f'槽位 {s} 需要 1+members*R={1 + a.members * a.R} 个核'
            Path(j['out']).mkdir(parents=True, exist_ok=True)
            cmd = ['taskset', '-c', s, str(Path(a.python).parent / 'mpirun'), '--bind-to', 'none', '-n', str(ranks), a.python, str(root / 'analysis/rbc_enkf_run.py'), '--truth', j['truth'], '--donor', j['donor'], '--out', j['out'], '--R', str(a.R), '--members', str(a.members)] + (['--device', f'cuda:{g}'] if g else []) + j['args']
            run[s] = (subprocess.Popen(cmd, stdout=open(Path(j['out']) / 'log.txt', 'a'), stderr=subprocess.STDOUT, env=env, cwd=root), j); print(time.strftime('%H:%M'), 'START', j['name'], 'slot', s, 'gpu', g, flush=True)
    time.sleep(20)
print('ALL FINISHED', flush=True)
