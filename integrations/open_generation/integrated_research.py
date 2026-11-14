"""DeltaLM optional generation interface and training feasibility research.

Uses the unmodified 2026-11-14 GitHub DeltaLM kernel. This script DOES NOT
call GPT inference, train GPT weights, or claim semantic visual/audio/video
understanding. Outputs reproducible numeric structure readouts and deterministic
media derived from actual traces. GPT-authored prose and separately generated
image assets can be added and identified separately.
"""
from __future__ import annotations
from pathlib import Path
import sys
_repo = Path(__file__).resolve().parents[2]
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here if (_here/'delta_core.py').exists() else _repo))
from dataclasses import asdict
import json, math, hashlib, wave, random, csv, os
import numpy as np
from delta_core import DeltaKernel,Substrate

BASE=Path(__file__).resolve().parent
OUT=BASE/'results'
OUT.mkdir(exist_ok=True)

NODES=['tide','stone','light','wind','shore','seed','shadow','echo']  # researcher labels, not model-generated
EDGES=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
       (0,3),(3,6),(6,1),(1,4)]

def make_world(seed:int=0,rewrites:bool=True):
    rng=np.random.default_rng(seed)
    m=DeltaKernel(Substrate(max_relations=72,max_contacts=210,enable_rewrites=rewrites))
    ids=[]
    for i,(a,b) in enumerate(EDGES):
        ids.append(m.add_relation(NODES[a],NODES[b],float(.35+.3*rng.random()),
                                  float(.045+.10*rng.random())))
    return m, ids

def input_schedule(step,style='tide'):
    if style=='tide': return {35:(0,.24),82:(4,-.18),133:(7,.27),210:(2,-.12)}.get(step)
    if style=='reverse':return {35:(2,-.12),82:(7,.27),133:(4,-.18),210:(0,.24)}.get(step)
    if style=='still':return None
    raise ValueError(style)

def readout(m):
    o=m.observe(); rs=m.state.relations; cs=m.state.contacts
    values=[float(r.expression()) for r in rs.values()]
    couplings=[float(c.coupling) for c in cs.values()]
    return {'step':o['clock'],'relations':len(rs),'contacts':len(cs),
      'births':o['births'],'deaths':o['deaths'],'programs_ever':o['programs_created'],
      'active_program_trees':len({r.program for r in rs.values()} | {c.program for c in cs.values()}),
      'activity_abs':float(sum(abs(x) for x in values)),
      'mean_coupling':float(np.mean(couplings)) if couplings else 0.,
      'coupling_abs':float(sum(abs(x) for x in couplings)),
      'node_positions':{k:[float(x),float(y)] for k,(x,y) in m.state.lenses.items()},
      'links':[{'id':r.identity,'src':r.source,'dst':r.target,
                'value':round(float(r.expression()),6),'generation':r.generation,
                'program':repr(r.program)} for r in rs.values()],
      'contact_rewrites':int(sum(c.rewrites for c in cs.values()))}

def series(seed=10,style='tide',rewrites=True,steps=320,snapshot_step=160):
    m,ids=make_world(seed,rewrites)
    frames=[readout(m)]; snap=None
    for s in range(1,steps+1):
        pulse=input_schedule(s,style)
        if pulse:
            i,v=pulse;m.stimulate(ids[i],v)
        m.step()
        if s == snapshot_step: snap=m.snapshot()
        if s % 10==0:frames.append(readout(m))
    return {'seed':seed,'style':style,'rewrites':rewrites,
            'frames':frames,'snapshot_middle':snap,'final':readout(m)}

def visible_state_signature(o):
    return tuple(sorted((r['src'],r['dst'],r['generation'],r['program']) for r in o['links']))

def run_counterfactual(seed=10):
    m,ids=make_world(seed)
    for s in range(1,161):
        pulse=input_schedule(s,'tide')
        if pulse:m.stimulate(ids[pulse[0]],pulse[1])
        m.step()
    shared=m.snapshot()
    normal=DeltaKernel.from_snapshot(shared)
    frozen=DeltaKernel.from_snapshot(shared)
    frozen.base.enable_rewrites=False
    for s in range(161,321):
        pulse=input_schedule(s,'tide')
        if pulse:
            for w in (normal,frozen):w.stimulate(ids[pulse[0]],pulse[1])
        normal.step();frozen.step()
    a,b=readout(normal),readout(frozen)
    sa,sb=set(visible_state_signature(a)),set(visible_state_signature(b))
    return {'seed':seed,'normal':{k:v for k,v in a.items() if k not in ('node_positions','links')},
            'frozen':{k:v for k,v in b.items() if k not in ('node_positions','links')},
            'relation_signature_symmetric_difference':len(sa^sb),
            'contact_rewrite_delta':a['contact_rewrites']-b['contact_rewrites'],
            'shared_start':True,'frozen_change_is_external_intervention':True}

def train_readout():
    """Ridge on core-derived present-state features => 5-step future summaries.
    Group by entire world for split. Trains ONLY an external probe, not DeltaLM.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    def features(o):
        return [o[k] for k in ('relations','contacts','births','deaths','programs_ever',
                    'active_program_trees','activity_abs','mean_coupling','coupling_abs',
                    'contact_rewrites')]
    def targets(o):return [o[k] for k in ('relations','activity_abs','mean_coupling')]
    X,Y,groups,now=[],[],[],[]
    for seed in range(32):
        m,ids=make_world(100+seed)
        rng=np.random.default_rng(900+seed)
        # training interventions are observed before input state, no future information
        for t in range(165):
            if t in (25,51,90,115):m.stimulate(ids[int(rng.integers(0,len(ids)))],float(rng.choice([-.21,.21])))
            if t in (35,55,75,95,115,135,155):
                before=readout(m);sp=m.snapshot();n=DeltaKernel.from_snapshot(sp);n.step(5)
                X.append(features(before));Y.append(targets(readout(n)))
                groups.append(seed);now.append(targets(before))
            m.step()
    X=np.asarray(X);Y=np.asarray(Y);now=np.asarray(now);groups=np.asarray(groups)
    train=(groups<24);test=~train
    # scalers trained on train only; output scaled to compare different units
    model=make_pipeline(StandardScaler(),Ridge(alpha=4.0))
    model.fit(X[train],Y[train]); pred=model.predict(X[test]);base=now[test]
    scales=np.std(Y[train],axis=0)+1e-6
    maes=np.mean(np.abs(pred-Y[test]),axis=0)
    baseline=np.mean(np.abs(base-Y[test]),axis=0)
    result={'method':'External Ridge predictor of 5-step future state; NOT trained DeltaLM core',
       'worlds_train':24,'worlds_test':8,'train_samples':int(train.sum()),'test_samples':int(test.sum()),
       'targets':['relations','activity_abs','mean_coupling'],
       'ridge_absolute_MAE':maes.tolist(),'persistence_absolute_MAE':baseline.tolist(),
       'ridge_normalized_MAE':(maes/scales).tolist(),
       'persistence_normalized_MAE':(baseline/scales).tolist(),
       'learned_interface_is_optional':True,'joint_GPT_weight_training':False,
       'cross_world_split':True,
       'learned_parameters': {'scaler_mean':model.named_steps['standardscaler'].mean_.tolist(),
       'scaler_scale':model.named_steps['standardscaler'].scale_.tolist(),
       'ridge_coef':model.named_steps['ridge'].coef_.tolist(),
       'ridge_intercept':model.named_steps['ridge'].intercept_.tolist()}}
    return result

def record_all():
    raw=series(10,'tide'); reverse=series(10,'reverse'); still=series(10,'still')
    null=DeltaKernel();null.step(400)
    counter=[run_counterfactual(s) for s in (10,11,12)]
    # restart reproducibility: continue from same complete state
    same=DeltaKernel.from_snapshot(raw['snapshot_middle']);other=DeltaKernel.from_snapshot(raw['snapshot_middle'])
    for t in range(161,201):
        x=input_schedule(t,'tide')
        if x:
            for m in [same,other]: m.stimulate(f'r{x[0]}',x[1])
        same.step();other.step()
    snap_equal=(same.snapshot()==other.snapshot())
    # filename contains exactly what is and isn't provided
    trace={z['style']:{k:v for k,v in z.items() if k !='snapshot_middle'} for z in [raw,reverse,still]}
    (OUT/'four_media_core_trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2),encoding='utf-8')
    train=train_readout()
    (OUT/'external_training_probe.json').write_text(json.dumps(train,ensure_ascii=False,indent=2),encoding='utf-8')
    # Semantic descriptor mapping written by researcher and suitable for GPT prompt; no inventing semantics
    f=raw['frames']
    readings=[]
    for i in (0,4,8,12,16,20,24,28,32):
        x=f[i];prev=f[max(0,i-4)]
        activity_change=x['activity_abs']-prev['activity_abs']
        readings.append({'step':x['step'],'relations':x['relations'],
          'births_cumulative':x['births'],'deaths_cumulative':x['deaths'],
          'activity':round(x['activity_abs'],6),
          'direction':('stronger' if activity_change>.10 else 'weaker' if activity_change<-.10 else 'similar'),
          'interpretation_is_external':True})
    (OUT/'generation_control_trace.json').write_text(json.dumps({'initial_prompt':'A quiet shore, wind, stones, seeds, and an echo; no ending is prescribed.',
         'world_labels_supplied_by_researcher':NODES,'mapping_policy':'Only direction, relation birth/death, relative activity and observed contact structures; no automatic animal/game semantic inference',
         'readings':readings},indent=2),encoding='utf-8')
    ss=still['final'];fs=raw['final'];rv=reverse['final']
    out={'source':'unmodified DeltaLM core','core_expected_sha':'515bb96dece31d81ad8a55deaf9c2d13b937a96c',
       'media_trace_seed':10,'trace_end_step':320,
       'active_final':{k:v for k,v in fs.items() if k not in ['node_positions','links']},
       'reversed_final':{k:v for k,v in rv.items() if k not in ['node_positions','links']},
       'still_final':{k:v for k,v in ss.items() if k not in ['node_positions','links']},
       'zero_initial_state_after_400_steps':null.observe(),
       'counterfactuals':counter,'snapshots_restored_equivalently':snap_equal,
       'training_probe':train,
       'reproduction':{'python':'python integrated_research.py','seed':10}}
    (OUT/'assessment.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    return trace,out

if __name__=='__main__':
    tr,assess=record_all()
    print(json.dumps({k:assess[k] for k in ['active_final','reversed_final','still_final','zero_initial_state_after_400_steps','counterfactuals','snapshots_restored_equivalently','training_probe']},ensure_ascii=False,indent=2)[:10000])
