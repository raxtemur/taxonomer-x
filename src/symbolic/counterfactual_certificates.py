"""Exact structured max-marginal diagnostics for maximum-branching decisions.

For a selected choice, the exclusion gap is the loss in the globally optimal
objective when that choice is forbidden.  For an unselected candidate, the
inclusion gap is the loss when that candidate is forced.  These are global
quantities: all other parent choices are re-optimized after the intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from src.symbolic.maximum_branching import (
    ROOT,
    BranchCandidate,
    BranchingResult,
    maximum_branching,
)


@dataclass(frozen=True)
class DecisionCertificate:
    child: str
    selected_parent: str
    selected_exclusion_gap: float
    boundary_parent: str | None
    boundary_inclusion_gap: float | None
    optimum_score: float

    @property
    def selected_radius(self) -> float:
        """Largest adverse one-edge perturbation certified not to change selection."""
        return self.selected_exclusion_gap


def _solve(
    nodes: set[str],
    candidates: list[BranchCandidate],
    root_scores: dict[str, float],
    *,
    forbid: tuple[str, str] | None = None,
    force: tuple[str, str] | None = None,
) -> BranchingResult | None:
    if forbid is not None and force is not None:
        raise ValueError("specify either forbid or force")
    work = list(candidates)
    allowed_roots = set(nodes)
    if forbid is not None:
        child, parent = forbid
        if parent == ROOT:
            allowed_roots.discard(child)
        else:
            work = [row for row in work if not (row.child == child and row.parent == parent)]
    if force is not None:
        child, parent = force
        work = [row for row in work if row.child != child]
        if parent == ROOT:
            pass
        else:
            allowed_roots.discard(child)
            forced = [row for row in candidates if row.child == child and row.parent == parent]
            if not forced:
                return None
            work.append(max(forced, key=lambda row: row.score))
    if not allowed_roots:
        return None
    try:
        result = maximum_branching(nodes, work, root_scores, allowed_roots=allowed_roots)
    except (ValueError, nx.NetworkXException):
        return None
    if force is not None and result.parents.get(force[0]) != force[1]:
        raise AssertionError("forced choice was not selected")
    if forbid is not None and result.parents.get(forbid[0]) == forbid[1]:
        raise AssertionError("forbidden choice remained selected")
    return result


def exact_decision_certificates(
    nodes: list[str] | set[str],
    candidates: list[BranchCandidate],
    root_scores: dict[str, float],
    boundary_by_child: dict[str, str] | None = None,
) -> tuple[BranchingResult, dict[str, DecisionCertificate]]:
    """Compute selected exclusion and optional boundary inclusion gaps exactly."""
    vocabulary = set(nodes)
    optimum = maximum_branching(vocabulary, candidates, root_scores)
    boundary_by_child = boundary_by_child or {}
    rows: dict[str, DecisionCertificate] = {}
    for child in sorted(vocabulary):
        selected = optimum.parents[child]
        excluded = _solve(vocabulary, candidates, root_scores, forbid=(child, selected))
        exclusion_gap = math.inf if excluded is None else max(0.0, optimum.score - excluded.score)
        boundary = boundary_by_child.get(child)
        inclusion_gap: float | None = None
        if boundary is not None:
            if boundary == selected:
                inclusion_gap = 0.0
            else:
                included = _solve(vocabulary, candidates, root_scores, force=(child, boundary))
                inclusion_gap = math.inf if included is None else max(0.0, optimum.score - included.score)
        rows[child] = DecisionCertificate(
            child=child,
            selected_parent=selected,
            selected_exclusion_gap=exclusion_gap,
            boundary_parent=boundary,
            boundary_inclusion_gap=inclusion_gap,
            optimum_score=optimum.score,
        )
    return optimum, rows


def certificate_features(rows: dict[str, DecisionCertificate], ordered_nodes: list[str]) -> list[dict[str, float]]:
    """Historical alias returning finite sensitivity features for an allocation gate."""
    finite = [
        value
        for row in rows.values()
        for value in (row.selected_exclusion_gap, row.boundary_inclusion_gap)
        if value is not None and math.isfinite(value)
    ]
    cap = max(finite, default=1.0) + 1.0
    output = []
    for node in ordered_nodes:
        row = rows[node]
        exclusion = row.selected_exclusion_gap if math.isfinite(row.selected_exclusion_gap) else cap
        raw_inclusion = row.boundary_inclusion_gap
        inclusion = raw_inclusion if raw_inclusion is not None and math.isfinite(raw_inclusion) else cap
        output.append({
            "global_selected_exclusion_gap": float(exclusion),
            "global_boundary_inclusion_gap": float(inclusion),
            "global_selected_fragility": 1.0 / (1.0 + float(exclusion)),
            "global_boundary_competitiveness": 1.0 / (1.0 + float(inclusion)),
            "global_certificate_pressure": 1.0 / (1.0 + float(exclusion) + float(inclusion)),
        })
    return output
