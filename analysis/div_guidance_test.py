"""Quick check of the DPS incompressibility guidance (--div-weight) on the validation case: python analysis/div_guidance_test.py WEIGHT MEMBERS DEVICE STEPS [PRIOR TAG]
Writes outputs/study_rb_div/full_w<W>_m<M>_n<STEPS>/ and prints RMSE and raw divergence (obs noise 0.1, every frame)."""
import sys, argparse, numpy as np, pandas as pd
sys.path.insert(0,'analysis')
import study_rb as S
from rdda.cycle import run_cycle
w=float(sys.argv[1]); mem=int(sys.argv[2]); dev=sys.argv[3]; steps=int(sys.argv[4])
prior=sys.argv[5] if len(sys.argv)>5 else 'outputs/rb_prior_4gpu/best.pt'; tag=('_'+sys.argv[6]) if len(sys.argv)>6 else ''
a=argparse.Namespace(data='data/processed',split='val',case_index=0,prior=prior,out='x',start_frame=0,analysis_steps=steps,rayleigh=1e5,model_rayleigh=5e4,
 observation_fraction=.01,members=mem,dps_steps=100,obs_std=float(__import__('os').environ.get('RDDA_OBS_STD',.05)),bg_std=.3,dps_rate=float(__import__('os').environ.get('RDDA_RATE',2.0)),init='noise',t_start=.4,hard_observation=False,div_weight=w,obs_noise=.1,every=1,seed=int(__import__('os').environ.get('RDDA_SEED',42)),device=dev)
truth,mask,fc=S.setup(a)
d=f'outputs/study_rb_div/full_w{w:g}_m{mem}_n{steps}{tag}'+(f"_s{a.seed}" if a.seed!=42 else ''); import os; os.makedirs(d,exist_ok=True)
res=run_cycle(truth,a.prior,fc,mask,analysis_steps=steps,method='bg_obs',assim_every=1,out=d,**S.cyc_kw(a),**S.dps_kw(a,obs_soft=float(__import__('os').environ.get('RDDA_SOFT',0))))
rows=S.enrich(res,d,fc,truth,a.std); df=pd.DataFrame(rows); df.to_csv(d+'/m.csv',index=False)
print('RESULT w',w,'members',mem,'rmse %.4f'%df.missing_rmse.mean(),'raw div %.4f'%df.an_div_rel.mean(),flush=True)
