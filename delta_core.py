"""DeltaLM research kernel: stateful relations, local conditions, rewritable operations.

No training objective, oracle, user couplings, or domain semantics in this core.
New programs are compositions of finite predeclared primitives, NOT arbitrary new math.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import Counter
from math import sin, cos, tanh, pi, isfinite
from typing import Any


def clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    if not isfinite(x):
        raise FloatingPointError('nonfinite state')
    return max(lo, min(hi, x))

# A tiny typed expression algebra, stored as immutable trees (tuples).
# Its primitives are *fixed*; effective executable structures need not be.
X, Y, H = ('x',), ('y',), ('h',)
BASE_OPERATORS = {X, Y, H}


def size(tree: tuple) -> int:
    return 1 + sum(size(child) for child in tree[1:])


def depth(tree: tuple) -> int:
    return 1 + max([depth(child) for child in tree[1:]] or [0])


def eval_tree(tree: tuple, x: float, y: float, history: float) -> float:
    op = tree[0]
    if op == 'x': return clip(x)
    if op == 'y': return clip(y)
    if op == 'h': return clip(history)
    vals = [eval_tree(ch, x, y, history) for ch in tree[1:]]
    if op == 'add': return clip(0.5 * (vals[0] + vals[1]))
    if op == 'sub': return clip(0.5 * (vals[0] - vals[1]))
    if op == 'mul': return clip(vals[0] * vals[1])
    if op == 'tanh': return tanh(2 * vals[0])
    if op == 'neg': return -vals[0]
    raise ValueError(f'unknown primitive: {op}')


def compress_tree(tree: tuple, max_depth: int = 5, max_size: int = 23) -> tuple:
    """Capacity-limited pruning; not mathematically lossless compression."""
    if depth(tree) <= max_depth and size(tree) <= max_size:
        return tree
    if len(tree) == 1:
        return tree
    kids = [compress_tree(k, max_depth - 1, max_size // 2) for k in tree[1:]]
    result = (tree[0], *kids)
    if max_depth <= 1 or size(result) > max_size:
        return min(kids, key=size) if kids else X
    return result


def join_program(left: tuple, right: tuple, condition: float, phase: float) -> tuple:
    """Joint state changes which program *structure* is written to the contact."""
    if condition < -0.06:
        out = ('sub', left, right)
    elif phase < -0.35:
        out = ('add', left, ('neg', right))
    elif condition > 0.10:
        out = ('mul', left, right)
    elif phase > 0.35:
        out = ('tanh', ('add', left, right))
    else:
        out = ('add', left, right)
    return compress_tree(out)


def decode_tree(obj: Any) -> tuple:
    if not isinstance(obj, (tuple, list)) or not obj or not isinstance(obj[0], str):
        raise ValueError('invalid expression tree')
    z = (obj[0], *(decode_tree(ch) for ch in obj[1:]))
    # Evaluate against a valid input to check primitive name and arity before use.
    arities = {'x':0, 'y':0, 'h':0, 'add':2, 'sub':2, 'mul':2, 'tanh':1, 'neg':1}
    if z[0] not in arities or len(z)-1 != arities[z[0]] or depth(z)>16 or size(z)>1024:
        raise ValueError('invalid expression grammar or size')
    return z


@dataclass
class Relation:
    identity: str
    source: str
    target: str
    activity: float
    trace: float
    phase: float
    pace: float
    plasticity: float = 0.5
    program: tuple = X
    origin: tuple[str, str] | None = None
    age: int = 0
    generation: int = 0

    def expression(self) -> float:
        return clip(self.activity + 0.15 * self.trace * sin(self.phase))


@dataclass
class Contact:
    left: str
    right: str
    coupling: float = 0.0  # 'coupling' here = physical coupling; not user authorization
    trace: float = 0.0
    fatigue: float = 0.0
    plasticity: float = 0.5
    program: tuple = ('mul', X, Y)
    last_rewrite: int = -100
    last_birth: int = -100
    rewrites: int = 0


@dataclass
class Substrate:
    max_relations: int = 72
    max_contacts: int = 192
    activity_retention: float = 0.971
    trace_retention: float = 0.998
    contact_retention: float = 0.986
    min_nucleation: float = 0.029
    removal_floor: float = 0.003
    rewrite_period: int = 13
    birth_refractory: int = 13
    enable_rewrites: bool = True
    auto_meet: bool = True


@dataclass
class Field:
    relations: dict[str, Relation] = field(default_factory=dict)
    contacts: dict[tuple[str, str], Contact] = field(default_factory=dict)
    lenses: dict[str, tuple[float,float]] = field(default_factory=dict)
    clock: int = 0
    next_id: int = 0
    births: int = 0
    deaths: int = 0
    generated_programs: set[tuple] = field(default_factory=set)


class DeltaKernel:
    def __init__(self, substrate: Substrate | None = None):
        self.base = substrate or Substrate()
        self.state = Field()

    def add_relation(self, source: str, target: str, intensity: float = 0.65,
                     pace: float = 0.11, program: tuple | None = None,
                     origin: tuple[str,str] | None = None) -> str:
        s = self.state
        if len(s.relations) >= self.base.max_relations:
            raise RuntimeError('relation capacity reached')
        ident = f'r{s.next_id}'
        s.next_id += 1
        if program is None:
            program = (X if s.next_id % 2 else Y)
        program = decode_tree(program)
        generation = 0 if origin is None else 1 + max(
            (s.relations[p].generation for p in origin if p in s.relations), default=0)
        s.relations[ident] = Relation(ident, str(source), str(target),
            clip(intensity), clip(abs(intensity),0,1),
            (0.19*s.next_id)%(2*pi), clip(pace,-1,1), program=program,
            origin=origin,generation=generation)
        s.lenses.setdefault(str(source),(0.,0.))
        s.lenses.setdefault(str(target),(0.,0.))
        if self.base.auto_meet:
            self._meet_neighbors(ident)
        return ident

    def _meet_neighbors(self, rid: str) -> None:
        if len(self.state.contacts) >= self.base.max_contacts:
            return
        r = self.state.relations[rid]
        for sid,q in list(self.state.relations.items()):
            if sid == rid or len(self.state.contacts) >= self.base.max_contacts:
                break
            if r.target == q.source:
                self._contact(rid,sid)
            if q.target == r.source and len(self.state.contacts)<self.base.max_contacts:
                self._contact(sid,rid)

    def _contact(self, left: str, right: str) -> Contact:
        s = self.state
        r,q=s.relations[left],s.relations[right]
        if r.target != q.source:
            raise ValueError('incompatible relation endpoints')
        key=(left,right)
        if key not in s.contacts:
            if len(s.contacts)>=self.base.max_contacts:
                raise RuntimeError('contact capacity reached')
            s.contacts[key]=Contact(left,right)
        return s.contacts[key]

    def encounter(self, left: str, right: str, strength: float=0.15) -> None:
        c=self._contact(left,right)
        x=clip(strength,0,1)
        self.state.relations[left].activity=clip(self.state.relations[left].activity+x)
        self.state.relations[right].activity=clip(self.state.relations[right].activity+x)
        c.trace=clip(c.trace+0.025*x)

    def stimulate(self, rid: str, value: float) -> None:
        r=self.state.relations[rid]
        r.activity=clip(r.activity+value)
        r.trace=clip(r.trace+0.08*abs(value),0,1)

    def _rewrite(self, c: Contact, r: Relation, q: Relation, alignment: float) -> None:
        if not self.base.enable_rewrites:
            return
        if self.state.clock-c.last_rewrite<self.base.rewrite_period:
            return
        # Do not manufacture endless novelty from two effectively dead signals.
        if abs(r.expression())+abs(q.expression())+abs(c.trace)<0.01:
            return
        candidate=join_program(r.program,q.program,c.trace-c.fatigue,alignment)
        c.last_rewrite=self.state.clock
        if candidate!=c.program:
            c.program=candidate
            c.rewrites+=1
            self.state.generated_programs.add(candidate)

    def step(self, steps: int=1) -> None:
        if steps<0: raise ValueError('negative step count')
        for _ in range(steps):
            self._once()

    def _once(self) -> None:
        s,b=self.state,self.base
        s.clock+=1
        for r in tuple(s.relations.values()):
            r.age+=1
            r.phase=(r.phase+r.pace*(1+0.15*r.plasticity))%(2*pi)
            r.activity=clip(r.activity*b.activity_retention)
            r.trace=clip(r.trace*b.trace_retention,0,1)
            r.plasticity=clip(0.995*r.plasticity+0.005*abs(r.trace),0,1)
        births=[]
        for key,c in tuple(s.contacts.items()):
            r,q=s.relations.get(c.left),s.relations.get(c.right)
            if r is None or q is None:
                s.contacts.pop(key,None)
                continue
            vx=s.lenses.get(r.source,(0,0))[0]
            vy=s.lenses.get(q.target,(0,0))[0]
            alignment=cos(r.phase-q.phase+0.1*(vx-vy))
            self._rewrite(c,r,q,alignment)
            x,y=r.expression(),q.expression()
            # Operator is executable and modulates the local coupling itself.
            joint=eval_tree(c.program,x,y,c.trace)
            encounter=clip(0.58*x*y*alignment+0.42*joint*alignment)
            c.coupling=clip(b.contact_retention*c.coupling+0.075*tanh(encounter-c.fatigue))
            c.trace=clip(0.994*c.trace+0.052*encounter)
            c.fatigue=clip(0.98*c.fatigue+0.022*abs(encounter),0,1)
            c.plasticity=clip(0.99*c.plasticity+0.01*abs(c.trace-c.coupling),0,1)
            r.pace=clip(r.pace+0.001*c.coupling*y,-1,1)
            q.pace=clip(q.pace-0.001*c.coupling*x,-1,1)
            r.trace=clip(r.trace+0.008*c.coupling*y,0,1)
            q.trace=clip(q.trace+0.008*c.coupling*x,0,1)
            impulse=encounter*(c.coupling+0.5*c.trace)
            limit=b.min_nucleation*(1+c.fatigue)/(0.5+c.plasticity)
            if abs(impulse)>=limit and s.clock-c.last_birth>=b.birth_refractory:
                births.append((key,impulse,encounter,c.program))
        existing={(r.source,r.target,r.program) for r in s.relations.values()}
        for key,impulse,encounter,program in births:
            if len(s.relations)>=b.max_relations: break
            c=s.contacts.get(key)
            if c is None or key[0] not in s.relations or key[1] not in s.relations: continue
            r,q=s.relations[key[0]],s.relations[key[1]]
            record=(r.source,q.target,program)
            if record in existing: continue
            # If new relation joins a later encounter it executes its inherited program.
            initial=clip(impulse+0.21*encounter)
            self.add_relation(r.source,q.target,initial,(r.pace+q.pace)/2,
                              program,origin=key)
            existing.add(record)
            s.births+=1
            c.last_birth=s.clock
        for r in tuple(s.relations.values()):
            vx,vy=s.lenses.get(r.source,(0,0))
            s.lenses[r.source]=(clip(vx+0.002*r.expression()*cos(r.phase),-2,2),
                              clip(vy+0.002*r.expression()*sin(r.phase),-2,2))
        for rid,r in tuple(s.relations.items()):
            if r.origin is not None and abs(r.activity)<b.removal_floor and abs(r.trace)<b.removal_floor:
                del s.relations[rid]
                s.deaths+=1
        # No fixed global output criterion is being maximized.

    def observe(self) -> dict:
        s=self.state
        return {
            'clock':s.clock,'n_relations':len(s.relations),'n_contacts':len(s.contacts),
            'births':s.births,'deaths':s.deaths,
            'relations':[{'id':r.identity,'source':r.source,'target':r.target,
                          'value':r.expression(),'derived':r.origin is not None,
                          'generation':r.generation,
                          'program':r.program} for r in s.relations.values()],
            'contact_conditions':{'%s>%s'%k:c.coupling for k,c in s.contacts.items()},
            'programs_created':len(s.generated_programs),
            'lenses':dict(s.lenses),
        }

    def snapshot(self) -> dict:
        s=self.state
        return {
            'substrate':asdict(self.base),'clock':s.clock,'next_id':s.next_id,
            'births':s.births,'deaths':s.deaths,
            'relations':[asdict(r) for r in s.relations.values()],
            'contacts':[asdict(c) for c in s.contacts.values()],
            'lenses':{k:list(v) for k,v in s.lenses.items()},
            'generated_programs':sorted(s.generated_programs,key=repr),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> 'DeltaKernel':
        m=cls(Substrate(**data['substrate']))
        s=m.state
        s.clock=int(data['clock']);s.next_id=int(data['next_id'])
        s.births=int(data['births']);s.deaths=int(data['deaths'])
        for a in data['relations']:
            item=dict(a);item['program']=decode_tree(item['program'])
            if item.get('origin') is not None:item['origin']=tuple(item['origin'])
            r=Relation(**item);s.relations[r.identity]=r
        for a in data['contacts']:
            item=dict(a);item['program']=decode_tree(item['program'])
            c=Contact(**item);s.contacts[(c.left,c.right)]=c
        s.lenses={k:tuple(v) for k,v in data['lenses'].items()}
        s.generated_programs={decode_tree(p) for p in data['generated_programs']}
        return m