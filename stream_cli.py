"""Research-only state runner, no direct real-world actions or sensors."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from delta_core import DeltaKernel


def main():
    p=argparse.ArgumentParser(description='DeltaLM local evolution (research only)')
    p.add_argument('--state',type=Path,default=Path('delta_state.json'))
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('new');sub.add_parser('observe')
    add=sub.add_parser('add');add.add_argument('source');add.add_argument('target')
    add.add_argument('--intensity',type=float,default=.65)
    step=sub.add_parser('step');step.add_argument('n',type=int)
    meet=sub.add_parser('encounter');meet.add_argument('left');meet.add_argument('right')
    stim=sub.add_parser('stimulate');stim.add_argument('relation');stim.add_argument('value',type=float)
    args=p.parse_args()
    if args.command=='new':
        m=DeltaKernel()
    else:
        if not args.state.exists():p.error('state file does not exist; run new first')
        m=DeltaKernel.from_snapshot(json.loads(args.state.read_text()))
    if args.command=='add':
        print('new relation:',m.add_relation(args.source,args.target,args.intensity))
    elif args.command=='encounter':m.encounter(args.left,args.right)
    elif args.command=='stimulate':m.stimulate(args.relation,args.value)
    elif args.command=='step':m.step(args.n)
    if args.command!='observe':
        args.state.write_text(json.dumps(m.snapshot(),indent=2),encoding='utf8')
    print(json.dumps(m.observe(),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
