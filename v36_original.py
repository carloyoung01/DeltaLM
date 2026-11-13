"""DeltaLM minimal open-structure kernel (design draft, not behavior-tested).

No language-specific or task-success objective. A relation is a local stateful
entity; its contact with another relation may alter future contact conditions.
Domain-specific safety, authorization and input/output adapters are outside.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from math import sin, cos, tanh


def bound(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


@dataclass
class Relation:
    identity: str
    source: str
    target: str
    activity: float = 0.0
    trace: float = 0.0
    phase: float = 0.0
    pace: float = 0.1
    plasticity: float = 0.5
    origin: tuple[str, str] | None = None

    def expression(self) -> float:
        # Observable current activity is not the relation's entire state.
        return bound(self.activity + 0.15 * self.trace * sin(self.phase))


@dataclass
class Contact:
    left: str
    right: str
    permission: float = 0.0  # Local effective condition, not authorization.
    trace: float = 0.0
    fatigue: float = 0.0
    plasticity: float = 0.5


@dataclass
class Field:
    """This is the descriptive state, not an exhaustive possible-world set."""
    relations: dict[str, Relation] = field(default_factory=dict)
    contacts: dict[tuple[str, str], Contact] = field(default_factory=dict)
    lenses: dict[str, tuple[float, float]] = field(default_factory=dict)
    clock: int = 0
    next_id: int = 0


@dataclass(frozen=True)
class Substrate:
    """Finite implementation limits, not the system's learned success goal."""
    max_relations: int = 128
    activity_retention: float = 0.967
    trace_retention: float = 0.997
    contact_retention: float = 0.985
    min_nucleation: float = 0.20
    removal_floor: float = 0.002


class DeltaKernel:
    def __init__(self, substrate: Substrate | None = None):
        self.base = substrate or Substrate()
        self.state = Field()

    def add_relation(self, source: str, target: str,
                     intensity: float = 0.25, pace: float = 0.1) -> str:
        """One possible sensory adapter: provide a directly observed relation."""
        if len(self.state.relations) >= self.base.max_relations:
            raise RuntimeError("resource capacity reached")
        ident = f"r{self.state.next_id}"
        self.state.next_id += 1
        self.state.relations[ident] = Relation(
            ident, source, target, bound(intensity), bound(abs(intensity)),
            pace=pace,
        )
        self.state.lenses.setdefault(source, (0.0, 0.0))
        self.state.lenses.setdefault(target, (0.0, 0.0))
        return ident

    def encounter(self, left: str, right: str) -> None:
        """Expose two relations to one another; do not prescribe a result."""
        self._get_contact(left, right)

    def _get_contact(self, left: str, right: str) -> Contact:
        r, s = self.state.relations[left], self.state.relations[right]
        if r.target != s.source:
            raise ValueError("Only composable adjacency is represented in this draft")
        key = (left, right)
        if key not in self.state.contacts:
            self.state.contacts[key] = Contact(left, right)
        return self.state.contacts[key]

    def step(self) -> None:
        """One internal transition; no input event and no success criterion."""
        s, b = self.state, self.base
        s.clock += 1
        # Local autonomous change: even an inactive relation can hold a trace.
        for r in list(s.relations.values()):
            r.phase += r.pace * (1 + 0.15 * r.plasticity)
            r.activity *= b.activity_retention
            r.trace *= b.trace_retention
            r.plasticity = bound(0.995 * r.plasticity + 0.005 * abs(r.trace), 0, 1)

        proposals: list[tuple[str, str, float]] = []
        for key, c in list(s.contacts.items()):
            r, q = s.relations.get(c.left), s.relations.get(c.right)
            if r is None or q is None:
                del s.contacts[key]
                continue
            # A changing observation geometry can alter subsequent interaction.
            xr, _ = s.lenses.get(r.source, (0.0, 0.0))
            xq, _ = s.lenses.get(q.target, (0.0, 0.0))
            alignment = cos(r.phase - q.phase + 0.1 * (xr - xq))
            encounter = r.expression() * q.expression() * alignment
            # Contact updates itself, and changes the future efficacy of contact.
            c.permission = bound(b.contact_retention * c.permission
                                 + 0.07 * tanh(encounter - c.fatigue))
            c.trace = bound(0.993 * c.trace + 0.05 * encounter)
            c.fatigue = bound(0.98 * c.fatigue + 0.025 * abs(encounter), 0, 1)
            c.plasticity = bound(0.99 * c.plasticity
                                 + 0.01 * abs(c.trace - c.permission), 0, 1)
            # Feedback changes each source relation's later effect.
            r.pace += 0.001 * c.permission * q.expression()
            q.pace -= 0.001 * c.permission * r.expression()
            r.trace = bound(r.trace + 0.008 * c.permission * q.expression())
            q.trace = bound(q.trace + 0.008 * c.permission * r.expression())
            # Nucleation criterion is history-dependent through local contact.
            impulse = encounter * (c.permission + 0.5 * c.trace)
            threshold = b.min_nucleation * (1 + c.fatigue) / (0.5 + c.plasticity)
            if abs(impulse) >= threshold:
                proposals.append((c.left, c.right, impulse))

        # No global winner or normalization; finite capacity is a physical limit.
        for left, right, impulse in proposals:
            if len(s.relations) >= b.max_relations:
                break
            r, q = s.relations[left], s.relations[right]
            # A relation produced by two relations can itself take part later.
            new_id = self.add_relation(r.source, q.target, impulse,
                                       pace=(r.pace + q.pace) / 2)
            s.relations[new_id].origin = (left, right)

        # Observation coordinates also drift with experienced effective relations.
        for r in s.relations.values():
            x, y = s.lenses.get(r.source, (0.0, 0.0))
            dx = 0.002 * r.expression() * cos(r.phase)
            dy = 0.002 * r.expression() * sin(r.phase)
            s.lenses[r.source] = (x + dx, y + dy)

        # Genuine loss is possible; dormant traces may also survive.
        for rid, r in list(s.relations.items()):
            if r.origin is not None and abs(r.activity) < b.removal_floor \
                    and abs(r.trace) < b.removal_floor:
                del s.relations[rid]
        # Contacts referencing dissolved relations are cleared next step.

    def observe(self) -> dict:
        """A non-mutating projection, not a complete picture of the world."""
        s = self.state
        return {
            "clock": s.clock,
            "relations": [
                {"id": r.identity, "ends": (r.source, r.target),
                 "expression": r.expression(), "origin": r.origin}
                for r in s.relations.values()
            ],
            "lenses": dict(s.lenses),
            "contact_conditions": {
                f"{a}|{b}": c.permission
                for (a, b), c in s.contacts.items()
            },
        }

    def snapshot(self) -> dict:
        """State export only; no claim of lossless real-world history."""
        s = self.state
        return {
            "clock": s.clock,
            "next_id": s.next_id,
            "relations": [asdict(r) for r in s.relations.values()],
            "contacts": [asdict(c) for c in s.contacts.values()],
            "lenses": {name: list(xy) for name, xy in s.lenses.items()},
            "substrate": asdict(self.base),
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> 'DeltaKernel':
        kernel = cls(Substrate(**data["substrate"]))
        s = kernel.state
        s.clock, s.next_id = data["clock"], data["next_id"]
        for item in data["relations"]:
            if item.get("origin") is not None:
                item = {**item, "origin": tuple(item["origin"])}
            r = Relation(**item)
            s.relations[r.identity] = r
        for item in data["contacts"]:
            c = Contact(**item)
            s.contacts[(c.left, c.right)] = c
        s.lenses = {name: tuple(xy) for name, xy in data["lenses"].items()}
        return kernel
