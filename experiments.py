"""Reproducible diagnostics (NOT optimizing DeltaLM for benchmark labels).
Run: python experiments.py --quick or python experiments.py
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from delta_core import DeltaKernel,Substrate,eval_tree
from adapters import (image_features,text_features,audio_features,
                      video_features,encode_input,make_audio,make_video,text_corpus)

OUT=Path(__file__).parent/'results'
OUT.mkdir(exist_ok=True)

def run_chain(rewrite:bool=True, seed:int=0, steps:int=500,permutation=False):
    rng=np.random.default_rng(seed)
    m=DeltaKernel(Substrate(enable_rewrites=rewrite))
    links=[('a','b'),('b','c'),('c','d'),('d','e'),('e','f')]
    if permutation: links=links[::-1]
    ids=[m.add_relation(a,b,float(rng.uniform(.57,.83)),
                        pace=float(rng.uniform(.05,.16))) for a,b in links]
    trace=[]
    for t in range(steps):
        if t<180 and t%15==0:
            for rid in ids:
                m.stimulate(rid,float(rng.uniform(.11,.23)))
        m.step()
        if t%max(steps//150,1)==0:
            trace.append({'t':t,'n':len(m.state.relations),'contacts':len(m.state.contacts),
                          'new':m.state.births,'gone':m.state.deaths,
                          'operators':len(m.state.generated_programs),
                          'total':sum(abs(r.expression()) for r in m.state.relations.values())})
    return m,trace


def semantically_distinct(programs):
    probes=[(x,y,h) for x in (-.8,-.2,.3,.9) for y in (-.7,.1,.6) for h in (-.4,.4)]
    out={}
    for p in programs:
        sig=tuple(round(eval_tree(p,*xyz),3) for xyz in probes)
        out.setdefault(sig,repr(p))
    return len(out)


def genealogy_and_effect():
    x,tx=run_chain(True,42,450)
    y,ty=run_chain(False,42,450)
    novel=x.state.generated_programs
    # Input events are identical. Realized difference includes descendant
    # rewrites and their structural effects, not just names of expressions.
    vx=np.zeros(6);vy=np.zeros(6)
    for m,out in [(x,vx),(y,vy)]:
        for r in m.state.relations.values():
            if r.source in 'abcdef' and r.target in 'abcdef':
                out['abcdef'.index(r.source)]+=r.expression()
    summary={
       'composed_programs_ever':len(novel),
       'functionally_distinct_on_probes':semantically_distinct(novel),
       'derived_relations_born':x.state.births,
       'derived_relations_ablated':y.state.births,
       'derived_active':sum(r.origin is not None for r in x.state.relations.values()),
       'ops_in_derived':len({r.program for r in x.state.relations.values() if r.origin}),
       'max_live_generation':max(r.generation for r in x.state.relations.values()),
       'causal_difference_l2':float(np.linalg.norm(vx-vy)),
       'operator_example':repr(next(iter(sorted(novel,key=repr)),None)),
       'resource_limits':{'relations':x.base.max_relations,'contacts':x.base.max_contacts},
    }
    return summary,tx,ty


def stress(quick=False):
    steps=2500 if quick else 12000
    seeds=[0,1,2] if quick else [0,1,2,3,4,5]
    records=[]
    for seed in seeds:
        t0=time.perf_counter()
        m,tr=run_chain(True,seed,steps,permutation=bool(seed%2))
        vals=[]
        for r in m.state.relations.values():
            vals.extend([r.activity,r.trace,r.phase,r.pace,r.plasticity])
        for c in m.state.contacts.values():
            vals.extend([c.coupling,c.trace,c.fatigue,c.plasticity])
        records.append({'seed':seed,'steps':steps,'seconds':time.perf_counter()-t0,
          'finite':bool(np.all(np.isfinite(vals))),
          'within_capacity':len(m.state.relations)<=m.base.max_relations and len(m.state.contacts)<=m.base.max_contacts,
          'relations':len(m.state.relations),'contacts':len(m.state.contacts),
          'births':m.state.births,'deaths':m.state.deaths,
          'new_programs':len(m.state.generated_programs),'total_activity':tr[-1]['total']})
    return records


def snapshot_check():
    m,_=run_chain(True,7,90)
    copy=DeltaKernel.from_snapshot(json.loads(json.dumps(m.snapshot())))
    for i in range(75):
        m.step();copy.step()
    def canonical(m):
        z=m.snapshot();z['generated_programs']=sorted(z['generated_programs'],key=repr)
        return z
    return canonical(m)==canonical(copy)


def nearest_neighbor_accuracy(matrix, labels):
    d=cdist(matrix,matrix,'euclidean')
    np.fill_diagonal(d,np.inf)
    idx=np.argmin(d,axis=1)
    return float(np.mean(np.asarray(labels)[idx]==np.asarray(labels)))


def dataset_modality(which,quick=False):
    rng=np.random.default_rng(123)
    samples=[]
    if which=='image':
        digits=load_digits()
        per=8 if quick else 15
        for k in range(10):
            inds=np.where(digits.target==k)[0]
            for i in inds[:per]: samples.append((image_features(digits.images[i]),k))
    elif which=='text':
        per=10 if quick else 20
        for txt,k in text_corpus():
            if sum(l==k for _,l in samples)<per:
                samples.append((text_features(txt),k))
    elif which=='audio':
        per=7 if quick else 12
        for k in range(6):
            for j in range(per):samples.append((audio_features(make_audio(k,j,rng)),k))
    elif which=='video':
        per=7 if quick else 12
        for k in range(4):
            for j in range(per):samples.append((video_features(make_video(k,j)),k))
    else:raise ValueError(which)
    raw=np.array([a for a,_ in samples]);labels=np.array([k for _,k in samples])
    record={'domain':which,'n_samples':len(samples),'n_classes':len(set(labels.tolist())),
            'input_adapter':'handcrafted; no end-to-end raw perception',
            'raw_1nn':nearest_neighbor_accuracy(raw,labels)}
    matrices={}
    for rewrites in (True,False):
        results=[encode_input(feat,rewrites,steps=7) for feat in raw]
        mat=np.array([r['vector'] for r in results]);
        tag='evolving' if rewrites else 'fixed_operator'
        matrices[tag]=mat
        record[tag+'_1nn']=nearest_neighbor_accuracy(mat,labels)
        record[tag+'_born_mean']=float(np.mean([r['born'] for r in results]))
        record[tag+'_programs_mean']=float(np.mean([r['operators'] for r in results]))
    record['mean_embedding_difference']=float(np.mean(np.linalg.norm(
        matrices['evolving']-matrices['fixed_operator'],axis=1)))
    return record


def v36_baseline():
    from v36_original import DeltaKernel as V36
    m=V36()
    for a,b in [('a','b'),('b','c'),('c','d'),('d','e'),('e','f')]:
        m.add_relation(a,b,intensity=.65)
    for i in range(250):m.step()
    return {'steps':250,'relations_final':len(m.state.relations),
            'derived_final':sum(r.origin is not None for r in m.state.relations.values()),
            'contacts':len(m.state.contacts),
            'note':'v36 requires explicit encounter; direct add_relation does not auto-connect'}


def compare_snapshot_counterfactual():
    """Change ONLY local executable programs, holding history fixed."""
    from delta_core import X,Y
    original,_=run_chain(True,9,110)
    clone=DeltaKernel.from_snapshot(json.loads(json.dumps(original.snapshot())))
    for c in clone.state.contacts.values():
        c.program=('mul',X,Y)
    # The second branch remains able to rewrite later; disable further rewrites
    # for both so the intervention is not immediately overwritten.
    original.base.enable_rewrites=False
    clone.base.enable_rewrites=False
    def profile(model):
        return np.array([sum(r.expression() for r in model.state.relations.values()
                              if r.source==chr(97+i) and r.target==chr(97+j))
                         for i in range(6) for j in range(6)]+[
                  sum(c.coupling for c in model.state.contacts.values()),
                  sum(abs(r.expression()) for r in model.state.relations.values())])
    before=np.linalg.norm(profile(original)-profile(clone))
    original.step(100)
    clone.step(100)
    return {'pre_intervention_observed_difference':float(before),
            'after_100_steps_difference_l2':float(np.linalg.norm(profile(original)-profile(clone))),
            'same_seed_and_prior_history':True}


def order_dependence():
    models=[]
    for ordering in ([0,1,2,3,4],[4,3,2,1,0]):
        m=DeltaKernel(Substrate(enable_rewrites=True))
        ids=[m.add_relation(str(i),str(i+1),.35,pace=.10) for i in range(5)]
        for k in ordering:
            m.stimulate(ids[k],.54)
            m.step(3)
        m.step(90)
        z=np.array([sum(r.expression() for r in m.state.relations.values()
                            if r.source==str(i) and r.target==str(j))
                    for i in range(6) for j in range(6)])
        models.append((m,z))
    return {'same_events_different_order_l2':float(np.linalg.norm(models[0][1]-models[1][1])),
            'births_forward':models[0][0].state.births,
            'births_reverse':models[1][0].state.births}


def continuing_input_stability():
    rows=[]
    for seed in (0,1,2):
        rng=np.random.default_rng(seed)
        m=DeltaKernel(Substrate(max_relations=72,max_contacts=192))
        ids=[m.add_relation(str(i),str(i+1),float(rng.uniform(.3,.8))) for i in range(6)]
        peak_n=0; peak_c=0
        for t in range(16000):
            if t%45==0:
                # Repeatedly encounter changing stimuli; no success objective.
                for idx in rng.choice(len(ids),size=3,replace=False):
                    m.stimulate(ids[int(idx)],float(rng.uniform(-.4,.5)))
            m.step()
            peak_n=max(peak_n,len(m.state.relations))
            peak_c=max(peak_c,len(m.state.contacts))
        values=[]
        for r in m.state.relations.values():
            values.extend((r.activity,r.trace,r.phase,r.pace,r.plasticity))
        for c in m.state.contacts.values():
            values.extend((c.coupling,c.trace,c.fatigue,c.plasticity))
        rows.append({'seed':seed,'steps':16000,'finite':bool(np.all(np.isfinite(values))),
                     'peak_relations':peak_n,'peak_contacts':peak_c,
                     'relations_final':len(m.state.relations),
                     'births':m.state.births,'deaths':m.state.deaths,
                     'programs_ever':len(m.state.generated_programs)})
    return rows


def make_figures(t1,t0,stress_result):
    fig,axs=plt.subplots(3,1,figsize=(10,9),sharex=True)
    for ax,col,label in zip(axs,('n','operators','total'),('Relations','New executable programs','Total absolute activity')):
        ax.plot([r['t'] for r in t1],[r[col] for r in t1],label='Rewrite enabled')
        ax.plot([r['t'] for r in t0],[r[col] for r in t0],label='Fixed operator',ls='--')
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axs[-1].set_xlabel('Internal steps');axs[0].legend()
    fig.tight_layout();fig.savefig(OUT/'operator_growth.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar([str(r['seed']) for r in stress_result],[r['relations'] for r in stress_result])
    ax.set_xlabel('History seed');ax.set_ylabel('Relations after long evolution')
    ax.set_title('Long-run structures (not a success score)');ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'long_run.png',dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();t0=time.perf_counter()
    structure,tx,ty=genealogy_and_effect()
    results={'run_kind':'quick' if args.quick else 'full',
             'genealogy':structure,'v36_baseline':v36_baseline(),
             'snapshot_resume_equivalent':snapshot_check(),
             'operator_intervention':compare_snapshot_counterfactual(),
             'history_order':order_dependence(),
             'ongoing_input_stability':continuing_input_stability(),
             'long_run':stress(args.quick),
             'modalities':[dataset_modality(x,args.quick) for x in ('image','text','audio','video')]}
    results['elapsed_seconds']=time.perf_counter()-t0
    make_figures(tx,ty,results['long_run'])
    (OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    with (OUT/'long_run.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results['long_run'][0].keys());w.writeheader();w.writerows(results['long_run'])
    with (OUT/'modalities.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=results['modalities'][0].keys());w.writeheader();w.writerows(results['modalities'])
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
