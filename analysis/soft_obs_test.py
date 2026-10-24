"""Single-step check of the observation weighting in DPS (no error accumulation). Background = Ra-mismatched solver run from the
TRUTH one frame earlier; observations = noisy truth (std 0.1). Grid over obs_std x soft x dps_rate; score = unobserved RMSE vs the
background's own error. python analysis/soft_obs_test.py DEVICE PRIOR OUT"""
import sys, itertools, argparse, numpy as np, pandas as pd
sys.path.insert(0,'analysis')
import study_rb as S
from rdda.assimilation import assimilate
dev, prior, out = sys.argv[1:4]
a=argparse.Namespace(data='data/processed',split='val',case_index=0,prior=prior,out='x',start_frame=0,analysis_steps=99,rayleigh=1e5,model_rayleigh=5e4,
 observation_fraction=.01,members=8,dps_steps=100,obs_std=.05,bg_std=.3,dps_rate=2.0,init='noise',t_start=.4,hard_observation=False,div_weight=0.,obs_noise=.1,every=1,seed=42,device=dev)
truth,mask,fc=S.setup(a); frames=list(range(10,100,6)); u=~mask
bgs={f: fc(truth[f-1], fc.steps_per_frame)[0] for f in frames}
rows=[]
for obs_std,soft,rate in itertools.product((0.05,0.1),(0.,1.,3.),(2.,4.,8.)):
    e=[]
    for f in frames:
        obs=S.noisy_observations(truth[f], mask, 0.1, a.seed, f)
        m=assimilate(prior, obs, mask, background=bgs[f], **S.dps_kw(a, seed=a.seed+f, obs_std=obs_std, dps_rate=rate, obs_soft=soft))['mean']
        e.append(S.metric(m,truth[f],u))
    rows.append(dict(obs_std=obs_std,soft=soft,rate=rate,an=float(np.mean(e)),bg=float(np.mean([S.metric(bgs[f],truth[f],u) for f in frames]))))
    print(rows[-1],flush=True); pd.DataFrame(rows).to_csv(out,index=False)
