"""Config-driven KPI driver tree.

The tree is declared in ``configs/kpi_tree.json`` (or ``.yaml`` if PyYAML is
installed). Two relation kinds are supported:

* ``product``: ``target = factor_1 x factor_2`` (e.g. active = paying x active_rate)
* ``flow``:    ``delta(target) = sum(inflows) - sum(outflows)`` (e.g. paying)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "kpi_tree.json"
KPI_TYPES = {"absolute", "accumulate", "rate_accumulate", "rate"}


@dataclass(frozen=True)
class Kpi:
    name: str
    label: str
    type: str
    unit: str = "merchants"
    good_direction: str = "up"
    numerator: str | None = None
    denominator: str | None = None


@dataclass(frozen=True)
class Relation:
    target: str
    kind: str
    factors: tuple[str, ...] = ()
    inflows: tuple[str, ...] = ()
    outflows: tuple[str, ...] = ()


@dataclass
class DriverTree:
    kpis: dict[str, Kpi]
    relations: list[Relation]
    headline_kpi: str
    dimensions: list[str] = field(default_factory=lambda: ["province", "channel"])
    mover_min_scale_share: float = 0.02

    def relation_for(self, target: str) -> Relation | None:
        return next((r for r in self.relations if r.target == target), None)

    def children(self, target: str) -> list[str]:
        r = self.relation_for(target)
        if r is None:
            return []
        return list(r.factors) if r.kind == "product" else list(r.inflows + r.outflows)

    def walk(self, root: str | None = None, depth: int = 0):
        """Yield ``(depth, kpi_name)`` in depth-first order from the root."""
        root = root or self.headline_kpi
        yield depth, root
        for c in self.children(root):
            yield from self.walk(c, depth + 1)


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml  # optional dependency, imported lazily

        return yaml.safe_load(text)
    return json.loads(text)


def load_tree(path: str | Path | None = None) -> DriverTree:
    raw = _read(Path(path) if path else DEFAULT_CONFIG)
    kpis = {}
    for name, spec in raw["kpis"].items():
        if spec["type"] not in KPI_TYPES:
            raise ValueError(f"KPI {name!r}: unknown type {spec['type']!r}")
        kpis[name] = Kpi(name=name, **spec)
    relations = []
    for r in raw.get("relations", []):
        rel = Relation(
            target=r["target"],
            kind=r["kind"],
            factors=tuple(r.get("factors", ())),
            inflows=tuple(r.get("inflows", ())),
            outflows=tuple(r.get("outflows", ())),
        )
        if rel.kind == "product" and len(rel.factors) != 2:
            raise ValueError("product relations need exactly two factors")
        for k in (rel.target, *rel.factors, *rel.inflows, *rel.outflows):
            if k not in kpis:
                raise ValueError(f"relation references unknown KPI {k!r}")
        relations.append(rel)
    return DriverTree(
        kpis=kpis,
        relations=relations,
        headline_kpi=raw["headline_kpi"],
        dimensions=list(raw.get("dimensions", ["province", "channel"])),
        mover_min_scale_share=float(raw.get("mover_min_scale_share", 0.02)),
    )
