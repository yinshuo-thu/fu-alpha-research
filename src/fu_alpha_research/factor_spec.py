from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


VIEWS = ("raw", "tsz", "csz", "csr")


@dataclass(frozen=True)
class FactorSpec:
    selected: list[str]
    by_view: dict[str, list[str]]
    needed_base: list[str]

    @property
    def n_factors(self) -> int:
        return len(self.selected)


def load_selected(path: str | Path) -> FactorSpec:
    selected = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    by_view: dict[str, list[str]] = {view: [] for view in VIEWS}
    needed: set[str] = set()
    for name in selected:
        for view in ("tsz", "csz", "csr"):
            prefix = f"{view}_"
            if name.startswith(prefix):
                base = name[len(prefix) :]
                by_view[view].append(base)
                needed.add(base)
                break
        else:
            by_view["raw"].append(name)
            needed.add(name)
    return FactorSpec(selected=selected, by_view=by_view, needed_base=sorted(needed))


def selected_subset(spec: FactorSpec, columns: list[str] | None) -> FactorSpec:
    if columns is None:
        return spec
    allowed = set(columns)
    selected = [c for c in spec.selected if c in allowed]
    by_view: dict[str, list[str]] = {view: [] for view in VIEWS}
    needed: set[str] = set()
    for name in selected:
        for view in ("tsz", "csz", "csr"):
            prefix = f"{view}_"
            if name.startswith(prefix):
                base = name[len(prefix) :]
                by_view[view].append(base)
                needed.add(base)
                break
        else:
            by_view["raw"].append(name)
            needed.add(name)
    return FactorSpec(selected=selected, by_view=by_view, needed_base=sorted(needed))
