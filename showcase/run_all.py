#!/usr/bin/env python3
"""DeltaLM Open Evolution Showcase v1.2: ten independent, falsifiable demonstrations.
Run from repository root: python showcase/run_all.py. Unmodified core, observer code separate.
No task rewards or evaluator outputs are fed into DeltaKernel. Deterministic conditions.
"""
from __future__ import annotations
import sys, json, math, hashlib, copy, csv, platform, time
from pathlib import Path
import numpy as np
import scipy,sklearn,skimage,matplotlib,PIL
from scipy.spatial.distance import cdist
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from skimage import data
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from delta_core import DeltaKernel,Substrate,eval_tree
from adapters import image_features,text_features,audio_features,make_audio,encode_input
OUT=ROOT/'showcase'/'results';OUT.mkdir(exist_ok=True,parents=True)
SEED=20261114

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')

def model(labels, seed=0,thresh=.022, max_relations=104, max_contacts=260,enable_rewrites=True):
    b=Substrate(max_relations=max_relations,max_contacts=max_contacts,min_nucleation=thresh,
                birth_refractory=11,rewrite_period=10,auto_meet=True,enable_rewrites=enable_rewrites)
    m=DeltaKernel(b);ids=[];n=len(labels)
    for i in range(n):
        a=.57+.20*math.sin((i+1)*(seed+1)*.73)
        p=.06+.11*math.sin((i+2)*(seed+2)*.41)
        ids.append(m.add_relation(labels[i],labels[(i+1)%n],a,p))
    for i in range(0,n,2):
        ids.append(m.add_relation(labels[i],labels[(i+2)%n],.27+.09*math.cos(i+seed),-.04+.01*i))
    return m,ids

def facts(m,t):
    s=m.state; rs=list(s.relations.values());cs=list(s.contacts.values())
    return dict(t=int(t),relations=len(rs),contacts=len(cs),births=s.births,deaths=s.deaths,
      operators=len(s.generated_programs),rewrites=sum(c.rewrites for c in cs),
      derived=sum(r.origin is not None for r in rs),
      generation=max((r.generation for r in rs),default=0),
      activity=round(sum(abs(r.expression()) for r in rs),8),
      enabled_pairs=sorted(f'{c.left}>{c.right}' for c in cs),
      programs=[repr(c.program) for c in cs[:8]])

def story(key,m,steps,events=None,every=5,checkpoints=()):
    events=events or {};frames=[];timeline=[];snapshots={}
    prev_rel=set(m.state.relations);prev_contact=set(m.state.contacts)
    prev_programs={k:repr(c.program) for k,c in m.state.contacts.items()}
    for t in range(steps+1):
        if t in events:
            for id_,val in events[t]:
                if id_ in m.state.relations:m.stimulate(id_,val)
            timeline.append({'t':t,'kind':'external_stimulation','entries':events[t]})
        f=facts(m,t);cr=set(m.state.relations);cc=set(m.state.contacts)
        f['new_relations']=len(cr-prev_rel);f['removed_relations']=len(prev_rel-cr)
        current_programs={k:repr(c.program) for k,c in m.state.contacts.items()}
        changed=sum(1 for k,v in current_programs.items() if k in prev_programs and prev_programs[k]!=v)
        f['program_changes']=changed
        if changed:timeline.append({'t':t,'kind':'contact_program_changed','count':changed})
        f['new_contacts']=len(cc-prev_contact);f['removed_contacts']=len(prev_contact-cc)
        if f['new_relations'] or f['removed_relations'] or f['new_contacts'] or f['removed_contacts']:
            timeline.append({'t':t,'kind':'observed_state_change','new_relations':f['new_relations'],
              'removed_relations':f['removed_relations'],'new_contacts':f['new_contacts'],
              'removed_contacts':f['removed_contacts']})
        if t%every==0 or t==steps:frames.append(f)
        if t in checkpoints:snapshots[str(t)]=m.snapshot()
        prev_rel=cr;prev_contact=cc;prev_programs=current_programs
        if t<steps:m.step()
    dump(f'{key}_trace.json',{'frames':frames,'events':timeline,'snapshots':snapshots})
    return m,frames,timeline

def audit_classification():
    # small but real held-out digits; plain toy word tasks and synthetic tones, all labels external.
    results={}
    for task in ['01_text','02_image','03_audio']:
        feats=[];labs=[];inputs=[]
        if task=='01_text':
            groups={'water':['boat','sea','harbor','tide','sail','port'],
                    'sky':['plane','wing','cloud','pilot','flight','airport'],
                    'food':['rice','chicken','market','meal','soup','spice'],
                    'weather':['rain','storm','wind','climate','sun','thunder']}
            for cls,(label,words) in enumerate(groups.items()):
                for j in range(18):
                    sentence=words[j%6]+' '+words[(j*3+2)%6]+' '+('near' if j%2 else 'today')
                    feats.append(text_features(sentence));labs.append(cls);inputs.append(sentence)
        elif task=='02_image':
            X,y=load_digits(return_X_y=True);rng=np.random.default_rng(SEED)
            chosen=[]
            for cls in range(10):chosen.extend(rng.choice(np.where(y==cls)[0],size=10,replace=False).tolist())
            for k in chosen:feats.append(image_features(X[k].reshape(8,8)));labs.append(int(y[k]));inputs.append(f'8x8_digits_index_{k}')
        else:
            rng=np.random.default_rng(SEED)
            for k in range(6):
                for j in range(9):
                    aud=make_audio(k,j,rng)
                    feats.append(audio_features(aud));labs.append(k);inputs.append(f'synthetic_tone_{k}_{j}')
        X=np.asarray(feats);y=np.asarray(labs)
        groups=[hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in X]
        # Avoid feature collision leakage. Group splits must have all labels train and test.
        from sklearn.model_selection import GroupShuffleSplit
        splits=[]
        for seed in range(SEED,SEED+400):
            tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.33,random_state=seed).split(X,y,groups))
            if set(y[tr])==set(y) and set(y[te])==set(y):splits.append((tr,te,seed))
            if len(splits)>=3:break
        embeds={'raw':X,'evolved':np.asarray([encode_input(x,True,7)['vector'] for x in X])}
        metrics={}
        for name,z in embeds.items():
            scores=[]
            for tr,te,_ in splits:
                pred=y[tr][cdist(z[te],z[tr]).argmin(axis=1)]
                scores.append(float(np.mean(pred==y[te])))
            metrics[name]={'scores':scores,'mean':float(np.mean(scores)) if scores else None}
        results[task]={'n':len(y),'classes':len(set(y)),'feature_unique':len(set(groups)),
          'splits':len(splits),'reading':'external nearest-neighbor classifier on predefined adapter features',
          'metrics':metrics,'input_examples':inputs[:6]}
    a=text_features('dog bites man').tolist();b=text_features('man bites dog').tolist()
    results['01_text']['word_order_collision']={'first':'dog bites man','second':'man bites dog',
      'identical_adapter_vectors':a==b,'first_features':a,'second_features':b}
    dump('01_03_references.json',results)
    return results

def visual_seed():
    # Public skimage sample is used at runtime; no third-party photograph redistributed.
    cat=data.chelsea();feat1=image_features(cat.mean(axis=2))
    # A predeclared synthetic checkerboard control. No claim that DeltaLM generated an animal.
    checker=np.indices((64,64)).sum(axis=0)%2
    feat2=image_features(checker)
    result=[]
    for label,f in [('cat_example_image',feat1),('synthetic_checker',feat2)]:
        m=DeltaKernel(Substrate(max_relations=80,max_contacts=190,min_nucleation=.024,
                            rewrite_period=10,birth_refractory=11))
        labels=[f'p{i}' for i in range(16)]
        ids=[m.add_relation(labels[i],labels[(i+1)%16],.18+.72*float(np.clip(v,0,1)),.07+.01*i) for i,v in enumerate(f)]
        m,frames,ev=story('04_'+label,m,150,every=3,checkpoints=[0,80,150])
        result.append({'label':label,'features':[round(float(x),6) for x in f],
             'final':frames[-1],'events':len(ev),'illustration':'relation graph, NOT generated animal morphology'})
    dump('04_image_seeds.json',result);return result

def open_world():
    records=[]
    for seed in [0,1,2]:
        labels=['water','stone','light','salt','root','shadow','current','shell']
        m,ids=model(labels,seed)
        acts={}
        if seed==0:acts={70:[(ids[0],.13)],130:[(ids[3],-.14)]}
        if seed==1:acts={95:[(ids[1],-.19)]}
        m,frames,events=story(f'05_world_seed_{seed}',m,270,acts,every=3,checkpoints=[0,130,270])
        records.append({'seed':seed,'initial_labels':labels,'predeclared_stimuli':acts,
           'final':frames[-1],'observed_events':len(events)})
    dump('05_world_summary.json',records);return records

def game():
    # SAME core with an author-defined action interface: no programmed goal/reward/win state.
    labels=['water','stone','wood','sound','light','mist','path']
    root,ids=model(labels,5)
    schema=[('look',0,None),('touch',20,(ids[2],.22)),('wait',40,None),('withdraw',70,(ids[2],-.18)),('return',105,(ids[0],.10))]
    events={t:[action] for _,t,action in schema if action is not None}
    continued,frames,history=story('06_game_actions',root,170,events,every=2,checkpoints=[0,40,100,170])
    root2,ids2=model(labels,5)
    _c,fr2,_ev=story('06_game_no_actions',root2,170,every=2)
    result={'user_actions_author_supplied':schema,'no_reward_or_win_state':True,
      'final_interactive':frames[-1], 'final_no_actions':fr2[-1],
      'contact_sets_differ':frames[-1]['enabled_pairs']!=fr2[-1]['enabled_pairs']}
    dump('06_game_summary.json',result);return result

def core07():
    records=[]
    for seed in (0,1,2):
        m,ids=model(['a','b','c','d','e','f','g','h'],seed,max_relations=105)
        m.step(12)  # start at shared complete state; no script-specific program tweaks
        snap=m.snapshot();normal=DeltaKernel.from_snapshot(snap);frozen=DeltaKernel.from_snapshot(snap)
        frozen.base.enable_rewrites=False # experimenter removes only future rewrite capability
        events={45:[(ids[1],-.20)],80:[(ids[5],.13)]}
        normal,na,na_events=story(f'07_normal_{seed}',normal,150,events,every=2,checkpoints=[0,150])
        frozen,nb,nb_events=story(f'07_frozen_{seed}',frozen,150,events,every=2,checkpoints=[0,150])
        common=set(normal.state.relations)&set(frozen.state.relations)
        l2=math.sqrt(sum((normal.state.relations[k].expression()-frozen.state.relations[k].expression())**2 for k in common))
        pA=set(normal.state.contacts);pB=set(frozen.state.contacts)
        def signatures(obj, with_program=False):
            ss=set()
            for c in obj.state.contacts.values():
                a=obj.state.relations.get(c.left);b=obj.state.relations.get(c.right)
                if a is None or b is None:continue
                sig=(a.source,a.target,b.target)
                if with_program:sig=(*sig,repr(c.program))
                ss.add(sig)
            return ss
        semantic_diff=len(signatures(normal)^signatures(frozen))
        signature_with_program_diff=len(signatures(normal,True)^signatures(frozen,True))
        rec={'seed':seed,'intervention':'freeze ONLY future Contact.program rewrites on a cloned full snapshot',
          'shared_initial_snapshot_sha256':hashlib.sha256(json.dumps(snap,sort_keys=True).encode()).hexdigest(),
          'full_initial_snapshots_identical':True,'initial_condition_change':{'enable_rewrites':False},
          'retained':na[-1],'frozen':nb[-1], 'shared_relation_expression_l2':l2,
          'contact_pair_symmetric_diff':len(pA^pB),
          'endpoint_signature_symmetric_diff':semantic_diff,
          'endpoint_plus_program_signature_symmetric_diff':signature_with_program_diff,
          'observed_contact_program_changes_retained':sum(e['count'] for e in na_events if e['kind']=='contact_program_changed'),
          'observed_contact_program_changes_frozen':sum(e['count'] for e in nb_events if e['kind']=='contact_program_changed')}
        records.append(rec)
    dump('07_counterfactual.json',records);return records

def coevolution08():
    labels=['a','b','c','d','e','f','g']
    base,ids=model(labels,1);snap=base.snapshot()
    cases={
      'no_events':{},'A_only':{15:[(ids[0],.28)]},'B_only':{35:[(ids[2],-.26)]},
      'A_then_B':{15:[(ids[0],.28)],35:[(ids[2],-.26)]},
      'B_then_A':{15:[(ids[2],-.26)],35:[(ids[0],.28)]},
      'A_B_simultaneous':{15:[(ids[0],.28),(ids[2],-.26)]}}
    r={};states={}
    for name,ev in cases.items():
        m=DeltaKernel.from_snapshot(snap)
        m,fr,events=story('08_'+name,m,135,ev,every=3)
        r[name]={'end':fr[-1],'history':ev}
        states[name]={k:rel.expression() for k,rel in m.state.relations.items()}
    shared=set.intersection(*(set(s) for s in states.values()))
    # difference from additive of A/B w.r.t no-event; nonlinear residual is NOT proof of unique emergent rule
    residual=math.sqrt(sum((states['A_then_B'][k]-states['A_only'][k]-states['B_only'][k]+states['no_events'][k])**2 for k in shared))
    ordered=math.sqrt(sum((states['A_then_B'][k]-states['B_then_A'][k])**2 for k in shared))
    dump('08_history_comparison.json',{'cases':r,'shared_relations':len(shared),
      'A_B_nonadditive_residual_l2':residual,'order_difference_l2':ordered,
      'note':'A numerical residual or noncommutativity alone is NOT proof of new emergent mechanism'})
    return {'nonadditivity':residual,'order':ordered,'cases':r}

def silence09():
    m,ids=model(['seed','light','air','stone','soil','water','shadow'],2,thresh=.024)
    m,fr,_=story('09_initial_activity',m,180,{50:[(ids[0],.27)]},every=5,checkpoints=[0,180])
    snap=m.snapshot();paused=DeltaKernel.from_snapshot(snap)
    paused,passive,_=story('09_passive',paused,1600,every=25,checkpoints=[0,300,1600])
    # returning external condition is a NEW intervention, NOT automatic return to former time
    stimulated=DeltaKernel.from_snapshot(paused.snapshot())
    stimulated,reactivation,_=story('09_recontact',stimulated,200,{0:[(ids[0],.30)]},every=5,checkpoints=[0,200])
    result={'activity_at_180':fr[-1]['activity'],'passive_final_activity':passive[-1]['activity'],
      'passive_births_since_fork':passive[-1]['births']-fr[-1]['births'],
      'passive_deaths_since_fork':passive[-1]['deaths']-fr[-1]['deaths'],
      'recontact_activity':reactivation[-1]['activity'],
      'snapshot_restorable':DeltaKernel.from_snapshot(snap).snapshot()==snap,
      'distinction':'player pause vs natural decay vs snapshot restore vs external recontact'}
    dump('09_persistence.json',result);return result

def ecology10():
    # A qualitative biology-inspired toy only. No genes/fitness/population dynamics or empirical prediction.
    labels=['pale','dark','habitat','sun','shade','prey','observer','season']
    base,ids=model(labels,3,thresh=.025);snap=base.snapshot()
    conditions={'stable':{},'light_shift':{35:[(ids[3],.27)],110:[(ids[4],-.21)]},
       'reverse_shift':{35:[(ids[4],.27)],110:[(ids[3],-.21)]}}
    r={}
    for name,ev in conditions.items():
        m=DeltaKernel.from_snapshot(snap)
        m,fr,events=story('10_'+name,m,210,ev,every=3,checkpoints=[0,110,210])
        r[name]={'conditions':ev,'end':fr[-1]}
    # deterministic *external* observer chooses one alive contact with highest existing |trace-coupling|, not an autonomous researcher.
    q=DeltaKernel.from_snapshot(snap);q.step(80)
    cs=list(q.state.contacts.values());selected=max(cs,key=lambda c:abs(c.trace-c.coupling)) if cs else None
    proposal={'source':'external deterministic selection from endogenous contact states, not self-directed planning',
       'contact':(selected.left,selected.right) if selected else None,
       'criterion':'max(abs(contact.trace - contact.coupling))'}
    dump('10_ecology_limits.json',{'toy_conditions':r,'external_structure_conditioned_proposal':proposal,
        'empirical_validation':'NOT PERFORMED: biology labels are author-supplied, no gene/fitness/ecology mechanism or withheld observed data'})
    return {'conditions':r,'proposal':proposal}

def main():
    ts=time.time();summary={'metadata':{'suite':'DeltaLM Open Evolution Showcase v1.2','run_seed':SEED,
        'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'scikit_learn':sklearn.__version__,
        'scikit_image':skimage.__version__,'matplotlib':matplotlib.__version__,
        'pillow':PIL.__version__,
        'core_sha256':hashlib.sha256((ROOT/'delta_core.py').read_bytes()).hexdigest(),
        'no_optimization_target':True,'external_features_and_visuals_are_not_model_generated':True}}
    for n,fn in [('01_03',audit_classification),('04',visual_seed),('05',open_world),('06',game),
                  ('07',core07),('08',coevolution08),('09',silence09),('10',ecology10)]:
        print('RUN',n,flush=True);start=time.time();summary[n]=fn();print('DONE',n,'seconds',round(time.time()-start,2),flush=True)
    summary['metadata']['runtime_seconds']=round(time.time()-ts,2)
    dump('summary.json',summary)
    print('FINAL SUMMARY',json.dumps({k:(v if k=='metadata' else 'written') for k,v in summary.items()},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
