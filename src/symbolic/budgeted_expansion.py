"""Budget-matched controls for lazy candidate expansion.

All policies share the same scored base prefix.  They differ only in how they
allocate the exact number of additional candidate pairs selected by the frozen
combined gate.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import networkx as nx

from src.symbolic.maximum_branching import BranchCandidate, ROOT, maximum_branching


PairRef = tuple[int, int]


@dataclass(frozen=True)
class GatePolicy:
    base_k: int = 10
    max_k: int = 20
    root_margin: float = 0.5
    edge_gap: float = 0.5

    def __post_init__(self) -> None:
        if self.base_k < 1 or self.max_k < self.base_k:
            raise ValueError("require 1 <= base_k <= max_k")
        if self.root_margin < 0 or self.edge_gap < 0:
            raise ValueError("margins must be non-negative")


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _extras_by_child(retrieved: list[list[tuple[int, float]]], policy: GatePolicy) -> dict[int, list[PairRef]]:
    return {
        child: [(child, rank) for rank in range(policy.base_k, min(policy.max_k, len(row)))]
        for child, row in enumerate(retrieved)
    }


def _take_budget(order: list[int], extras: dict[int, list[PairRef]], budget: int) -> list[PairRef]:
    selected: list[PairRef] = []
    for child in order:
        remaining = budget - len(selected)
        if remaining <= 0:
            break
        selected.extend(extras[child][:remaining])
    if len(selected) != budget:
        raise ValueError(f"could allocate only {len(selected)} of {budget} requested pairs")
    return selected


def _uniform_budget(extras: dict[int, list[PairRef]], budget: int) -> list[PairRef]:
    """Allocate rank-by-rank across children, with exact pair-count matching."""
    selected: list[PairRef] = []
    depth = max((len(row) for row in extras.values()), default=0)
    for offset in range(depth):
        for child in sorted(extras):
            if offset < len(extras[child]):
                selected.append(extras[child][offset])
                if len(selected) == budget:
                    return selected
    if len(selected) != budget:
        raise ValueError(f"could allocate only {len(selected)} of {budget} requested pairs")
    return selected


def diagnose_base(
    nodes: list[str],
    retrieved: list[list[tuple[int, float]]],
    scores: dict[PairRef, float],
    root_scores: list[float],
    policy: GatePolicy,
) -> dict:
    """Compute frozen neural-uncertainty and symbolic-conflict diagnostics."""
    if len(nodes) != len(retrieved) or len(nodes) != len(root_scores):
        raise ValueError("nodes, retrieved and root_scores must align")
    local_parents: dict[str, str] = {}
    local_choice_score: dict[int, float] = {}
    neural_uncertainty: dict[int, float] = {}
    root_edge_margin: dict[int, float] = {}
    edge_gap: dict[int, float] = {}
    candidates: list[BranchCandidate] = []
    for child, row in enumerate(retrieved):
        available = [
            (rank, scores[(child, rank)])
            for rank in range(min(policy.base_k, len(row)))
        ]
        for rank, score in available:
            candidates.append(BranchCandidate(nodes[child], nodes[row[rank][0]], score))
        ordered = sorted(available, key=lambda item: (-item[1], item[0]))
        best_edge = ordered[0][1] if ordered else -math.inf
        second_edge = ordered[1][1] if len(ordered) > 1 else -math.inf
        root_gap = abs(root_scores[child] - best_edge) if ordered else 0.0
        top_gap = best_edge - second_edge if len(ordered) > 1 else math.inf
        root_edge_margin[child] = root_gap
        edge_gap[child] = top_gap
        neural_uncertainty[child] = min(root_gap, top_gap)
        if not ordered or root_scores[child] >= best_edge:
            local_parents[nodes[child]] = ROOT
            local_choice_score[child] = root_scores[child]
        else:
            best_rank = ordered[0][0]
            local_parents[nodes[child]] = nodes[row[best_rank][0]]
            local_choice_score[child] = best_edge

    local_graph = nx.DiGraph()
    local_graph.add_nodes_from(nodes)
    local_graph.add_edges_from(
        (child, parent) for child, parent in local_parents.items() if parent != ROOT
    )
    cycle_nodes = {
        node
        for component in nx.strongly_connected_components(local_graph)
        if len(component) > 1
        for node in component
    }
    exact = maximum_branching(
        nodes,
        candidates,
        {node: float(root_scores[index]) for index, node in enumerate(nodes)},
    )
    exact_edit: dict[int, bool] = {}
    regret: dict[int, float] = {}
    for child, node in enumerate(nodes):
        chosen = exact.parents[node]
        exact_edit[child] = chosen != local_parents[node]
        if chosen == ROOT:
            chosen_score = root_scores[child]
        else:
            chosen_score = max(
                scores[(child, rank)]
                for rank in range(min(policy.base_k, len(retrieved[child])))
                if nodes[retrieved[child][rank][0]] == chosen
            )
        regret[child] = max(0.0, local_choice_score[child] - chosen_score)

    active = {
        child
        for child, node in enumerate(nodes)
        if root_edge_margin[child] <= policy.root_margin
        or edge_gap[child] <= policy.edge_gap
        or node in cycle_nodes
        or exact_edit[child]
    }
    return {
        "active": active,
        "neural_uncertainty": neural_uncertainty,
        "root_edge_margin": root_edge_margin,
        "edge_gap": edge_gap,
        "cycle": {child: nodes[child] in cycle_nodes for child in range(len(nodes))},
        "exact_edit": exact_edit,
        "regret": regret,
        "base_result": exact,
    }


def allocate_budget_matched(
    *,
    key: str,
    seed: int,
    retrieved: list[list[tuple[int, float]]],
    diagnostics: dict,
    policy: GatePolicy,
) -> dict[str, list[PairRef]]:
    """Return combined, equal-budget controls, and fixed/full reference sets."""
    extras = _extras_by_child(retrieved, policy)
    combined = [ref for child in sorted(diagnostics["active"]) for ref in extras[child]]
    budget = len(combined)
    children = list(range(len(retrieved)))
    neural_order = sorted(
        children,
        key=lambda child: (diagnostics["neural_uncertainty"][child], child),
    )
    symbolic_order = sorted(
        children,
        key=lambda child: (
            -int(diagnostics["exact_edit"][child]),
            -int(diagnostics["cycle"][child]),
            -diagnostics["regret"][child],
            child,
        ),
    )
    random_order = children.copy()
    random.Random(_stable_seed(seed, key)).shuffle(random_order)
    full = [ref for child in children for ref in extras[child]]
    return {
        "fixed_k10": [],
        "combined": combined,
        "neural_only": _take_budget(neural_order, extras, budget),
        "symbolic_only": _take_budget(symbolic_order, extras, budget),
        "random": _take_budget(random_order, extras, budget),
        "uniform_budget": _uniform_budget(extras, budget),
        "full_k20": full,
    }
