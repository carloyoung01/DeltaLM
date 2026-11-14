#!/usr/bin/env python3
"""Tiny command-driven open-world interface. No score/win/end rule.
The world is a deterministic DeltaLM state, not an LLM and not a game story generator.
Run from root: python showcase/play_world.py
"""
import sys,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from delta_core import DeltaKernel,Substrate

def new_world():
    m=DeltaKernel(Substrate(max_relations=104,max_contacts=260,min_nucleation=.022,
        rewrite_period=10,birth_refractory=11))
    names=['water','stone','wood','sound','light','mist','path']
    for i,x in enumerate(names):m.add_relation(x,names[(i+1)%len(names)],.6+.1*((i%3)-1),.07+.02*i)
    for i in range(0,len(names),2):m.add_relation(names[i],names[(i+2)%len(names)],.3,.04)
    return m

def status(m):
    o=m.observe(); print(f"step={o['clock']} relations={o['n_relations']} contacts={o['n_contacts']} births={o['births']} deaths={o['deaths']} operator_trees={o['programs_created']}")
    for r in o['relations'][:12]:print(f"  {r['id']} {r['source']}->{r['target']} value={r['value']:.4f} gen={r['generation']}")
    if len(o['relations'])>12:print(f"  ... {len(o['relations'])-12} more relations")

def main():
    m=new_world();print('DeltaLM Open World. Actions: look, step N, touch ID VALUE, save PATH, load PATH, quit. No win state.');status(m)
    while True:
        try:cmd=shlex.split(input('world> '))
        except (EOFError,KeyboardInterrupt):print();break
        if not cmd:continue
        try:
            if cmd[0]=='look':status(m)
            elif cmd[0]=='step':
                n=int(cmd[1]) if len(cmd)>1 else 1
                if not 0<=n<=10000:raise ValueError('steps must be between 0 and 10000')
                m.step(n);status(m)
            elif cmd[0]=='touch' and len(cmd)==3:
                rid=cmd[1];value=float(cmd[2]);
                if not -1<=value<=1:raise ValueError('strength must be in [-1,1]')
                m.stimulate(rid,value);status(m)
            elif cmd[0]=='save' and len(cmd)==2:
                p=Path(cmd[1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(m.snapshot()),encoding='utf8');print('saved',p)
            elif cmd[0]=='load' and len(cmd)==2:
                m=DeltaKernel.from_snapshot(json.loads(Path(cmd[1]).read_text(encoding='utf8')));status(m)
            elif cmd[0] in ('quit','exit'):break
            else:print('Unknown action; use look, step N, touch ID VALUE, save PATH, load PATH, quit')
        except (ValueError,KeyError,OSError,IndexError,RuntimeError) as exc:print('Action not applied:',exc)
if __name__=='__main__':main()