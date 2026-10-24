"""Single-step A (background as DPS guidance + observations) for several priors, NO error accumulation: the background is the
Ra-mismatched solver run from the TRUTH one frame earlier, observations are noisy truth; same background/obs/mask for every prior.
python analysis/single_step_prior_test.py DEVICE OUT PRIOR[:NAME] ... (validation case 0, frames 10..99 step 4, 32 samples)"""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0,'analysis')
import study_rb as S
from rdda.assimilation import assimilate
dev, out = sys.argv[1], sys.argv[2]; priors = [p.split(':') if ':' in p else [p, p] for p in sys.argv[3:]]
a=argparse.Namespace(data='data/processed',split='val',case_index=0,prior='x',out='x',start_frame=0,analysis_steps=99,rayleigh=1e5,model_rayleigh=5e4,
 observation_fraction=.01,members=32,dps_steps=100,obs_std=.05,bg_std=.3,dps_rate=2.0,init='noise',t_start=.4,hard_observation=False,div_weight=0.,obs_noise=.1,every=1,seed=42,device=dev)
truth,mask,fc=S.setup(a)
frames=list(range(10,100,4)); bgs={f: fc(truth[f-1], fc.steps_per_frame)[0] for f in frames}
rows=[]
for noise in (0.1, 0.0):
    for f in frames:
        xt=truth[f]; obs=S.noisy_observations(xt, mask, noise, a.seed, f); u=~mask
        for p,name in priors:
            for label,bg in (('bg+obs',bgs[f]),('obs only',None)):
                kw=S.dps_kw(a, seed=a.seed+f, **({'background':bg} if bg is not None else {'background':None}))
                m=assimilate(p, obs, mask, **kw)['mean']
                rows.append(dict(noise=noise,frame=f,prior=name,scheme=label,an=S.metric(m,xt,u),bg=S.metric(bgs[f],xt,u),
                                 corr_cos=float((((m-bgs[f])[u])*((xt-bgs[f])[u])).sum()/np.linalg.norm((m-bgs[f])[u])/np.linalg.norm((xt-bgs[f])[u]))))
        print('noise',noise,'frame',f,flush=True)
        pd.DataFrame(rows).to_csv(out,index=False)
d=pd.DataFrame(rows); print(d.groupby(['noise','scheme','prior'])[['an','bg','corr_cos']].mean().round(4).to_string())
