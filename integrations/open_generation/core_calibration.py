"""Synthetic supervised calibration of *existing* DeltaLM Substrate parameters.
This is not GPT-weight training or learned semantic decoding.
Run: python integrations/open_generation/core_calibration.py
"""
from __future__ import annotations
import sys,json,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution
REPO=Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:sys.path.insert(0,str(REPO))
from delta_core import DeltaKernel,Substrate

HERE=Path(__file__).resolve().parent
DEFAULT=(Substrate().activity_retention,Substrate().contact_retention)
TEACHER=(.943,.963)  # disclosed synthetic system, NOT real-world observations
TRAIN=(120,121,122,123)
TEST=(300,301,302)
STEPS=(5,10,18,28)

def trajectory(seed,values):
    rng=np.random.default_rng(seed)
    b=Substrate(max_relations=15,max_contacts=32,
        activity_retention=float(values[0]),contact_retention=float(values[1]))
    m=DeltaKernel(b)
    ns=('a','b','c','d','e','f')
    for i in range(6):
        m.add_relation(ns[i],ns[(i+1)%6],float(rng.uniform(.28,.68)),
                       float(rng.uniform(.055,.15)))
    interventions={t:(int(rng.integers(0,6)),float(rng.choice([-.14,.14])))
                   for t in (8,15,23)}
    out=[]
    for t in range(1,max(STEPS)+1):
        if t in interventions:
            idx,val=interventions[t]
            m.stimulate('r'+str(idx),val)
        m.step()
        if t in STEPS:
            signals=[m.state.relations['r'+str(i)].expression() for i in range(6)]
            cs=[c.coupling for c in m.state.contacts.values()]
            out.append(signals+[sum(abs(x) for x in cs)/max(1,len(cs))])
    return np.asarray(out,float)

def run(seed=42,maxiter=6):
    targets={s:trajectory(s,TEACHER) for s in TRAIN+TEST}
    def error(seeds,p):
        return float(np.mean([np.abs(trajectory(s,p)-targets[s]).mean() for s in seeds]))
    fit=differential_evolution(lambda v:error(TRAIN,v),
          bounds=[(.91,.991),(.925,.999)],seed=seed,maxiter=maxiter,
          popsize=4,polish=True,tol=.0005,atol=1e-7,workers=1)
    p=tuple(map(float,fit.x))
    return {
      'status':'completed_synthetic_parameter_training',
      'training_type':'derivative-free calibration of real DeltaLM Substrate coefficients',
      'teacher_synthetic':True,'teacher_parameters':list(TEACHER),
      'original_parameters':dict(zip(('activity_retention','contact_retention'),DEFAULT)),
      'learned_parameters':dict(zip(('activity_retention','contact_retention'),p)),
      'train_world_seeds':list(TRAIN),'heldout_world_seeds':list(TEST),
      'measured_steps':list(STEPS),
      'metric':'MAE of six original relation expressions plus mean absolute contact coupling',
      'training_mae_baseline':error(TRAIN,DEFAULT),
      'training_mae_learned':error(TRAIN,p),
      'heldout_mae_baseline':error(TEST,DEFAULT),
      'heldout_mae_learned':error(TEST,p),
      'optimizer_nfev':int(fit.nfev),'optimizer_success':bool(fit.success),
      'optimizer_message':str(fit.message),
      'cross_world_generalization':error(TEST,p)<error(TEST,DEFAULT),
      'GPT_weights_trained':False,'semantic_decoder_trained':False,
      'historical_archive_required_by_core':False,
      'scientific_limit':'Parameter identification for same synthetic simulator family only.'
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--maxiter',type=int,default=6)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    result=run(maxiter=args.maxiter)
    path=Path(args.output) if args.output else HERE/'results'/'core_calibration_2026-11-14.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
