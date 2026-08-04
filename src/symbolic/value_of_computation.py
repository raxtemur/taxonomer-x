"""Learned value-of-computation allocation from neural and structural diagnostics.

The gate is fitted only on development graphs. At test time it observes scores
for a fixed candidate prefix, ranks children, and allocates an exact number of
additional child--candidate pairs. Hard constraints remain in the downstream
exact decoder; the gate changes compute allocation, not the feasible family.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import networkx as nx
import torch

from src.symbolic.budgeted_expansion import (
    GatePolicy,
    PairRef,
    _extras_by_child,
    _take_budget,
    _uniform_budget,
)
from src.symbolic.maximum_branching import BranchingResult, ROOT


NEURAL_FEATURES = (
    "root_edge_uncertainty",
    "top_gap_uncertainty",
    "best_edge_score",
    "root_score",
    "retrieval_boundary_score",
    "retrieval_boundary_gap",
)
SYMBOLIC_FEATURES = (
    "in_cycle",
    "cycle_size_log",
    "fan_in_log",
    "edge_selected",
    "cycle_break_cost",
)
ALL_FEATURES = NEURAL_FEATURES + SYMBOLIC_FEATURES


@dataclass(frozen=True)
class RidgeGate:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    l2: float

    def as_dict(self) -> dict:
        return {
            "feature_names": list(self.feature_names),
            "mean": list(self.mean),
            "scale": list(self.scale),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "l2": self.l2,
        }


def _safe(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def local_gate_features(
    nodes: list[str],
    retrieved: list[list[tuple[int, float]]],
    scores: dict[PairRef, float],
    root_scores: list[float],
    policy: GatePolicy,
) -> list[dict[str, float]]:
    """Compute test-label-free features without running an exact base decode."""
    if not (len(nodes) == len(retrieved) == len(root_scores)):
        raise ValueError("nodes, retrieved, and root scores must align")
    parents: dict[int, int | None] = {}
    best_scores: dict[int, float] = {}
    rows: list[dict[str, float]] = []
    for child, candidates in enumerate(retrieved):
        available = [
            (rank, float(scores[(child, rank)]))
            for rank in range(min(policy.base_k, len(candidates)))
        ]
        ordered = sorted(available, key=lambda item: (-item[1], item[0]))
        best = ordered[0][1] if ordered else float(root_scores[child])
        second = ordered[1][1] if len(ordered) > 1 else best
        edge_gap = max(0.0, best - second)
        root_gap = abs(float(root_scores[child]) - best)
        if ordered and best > float(root_scores[child]):
            parents[child] = int(candidates[ordered[0][0]][0])
            best_scores[child] = best
        else:
            parents[child] = None
            best_scores[child] = float(root_scores[child])
        finite_retrieval_scores = [
            float(score) for _, score in candidates if math.isfinite(float(score))
        ]
        retrieval_floor = min(finite_retrieval_scores, default=0.0)
        boundary = min(policy.base_k, len(candidates)) - 1
        boundary_raw = float(candidates[boundary][1]) if boundary >= 0 else retrieval_floor
        boundary_score = _safe(boundary_raw, retrieval_floor)
        next_raw = (
            float(candidates[policy.base_k][1])
            if len(candidates) > policy.base_k
            else boundary_score
        )
        next_score = _safe(next_raw, retrieval_floor)
        rows.append({
            "root_edge_uncertainty": 1.0 / (1.0 + root_gap),
            "top_gap_uncertainty": 1.0 / (1.0 + edge_gap),
            "best_edge_score": _safe(best),
            "root_score": _safe(root_scores[child]),
            "retrieval_boundary_score": boundary_score,
            "retrieval_boundary_gap": max(0.0, boundary_score - next_score),
        })

    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes)))
    graph.add_edges_from((child, parent) for child, parent in parents.items() if parent is not None)
    cycle_size = {child: 0 for child in range(len(nodes))}
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            for child in component:
                cycle_size[child] = len(component)
    for child, row in enumerate(rows):
        selected_edge = parents[child] is not None
        row.update({
            "in_cycle": float(cycle_size[child] > 0),
            "cycle_size_log": math.log1p(cycle_size[child]),
            "fan_in_log": math.log1p(graph.in_degree(child)),
            "edge_selected": float(selected_edge),
            "cycle_break_cost": (
                abs(best_scores[child] - float(root_scores[child]))
                if cycle_size[child] > 0 and selected_edge
                else 0.0
            ),
        })
    return rows


def fit_ridge_gate(
    rows: list[dict[str, float]],
    targets: list[float],
    *,
    feature_names: tuple[str, ...],
    l2: float,
) -> RidgeGate:
    if len(rows) != len(targets) or not rows:
        raise ValueError("non-empty feature rows and targets must align")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    matrix = torch.tensor(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=torch.float64,
    )
    target = torch.tensor(targets, dtype=torch.float64)
    if not torch.isfinite(matrix).all():
        raise ValueError("gate features must all be finite")
    if not torch.isfinite(target).all():
        raise ValueError("gate targets must all be finite")
    mean = matrix.mean(dim=0)
    scale = matrix.std(dim=0, unbiased=False).clamp_min(1e-8)
    standardized = (matrix - mean) / scale
    design = torch.cat((standardized, torch.ones((len(rows), 1), dtype=torch.float64)), dim=1)
    penalty = torch.eye(design.shape[1], dtype=torch.float64) * l2
    penalty[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ target)
    return RidgeGate(
        feature_names=feature_names,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(float(value) for value in weights[:-1]),
        intercept=float(weights[-1]),
        l2=l2,
    )


def predict_gate(gate: RidgeGate, rows: list[dict[str, float]]) -> list[float]:
    output = []
    for row in rows:
        value = gate.intercept
        for name, mean, scale, weight in zip(
            gate.feature_names, gate.mean, gate.scale, gate.coefficients
        ):
            value += weight * (float(row[name]) - mean) / scale
        if not math.isfinite(value):
            raise ValueError("gate prediction must be finite")
        output.append(float(value))
    return output


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"voc:{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def extra_pair_budget(retrieved: list[list[tuple[int, float]]], policy: GatePolicy, pair_fraction: float) -> int:
    if not 0.5 <= pair_fraction <= 1.0:
        raise ValueError("pair_fraction must be in [0.5, 1.0]")
    base = sum(min(policy.base_k, len(row)) for row in retrieved)
    full = sum(min(policy.max_k, len(row)) for row in retrieved)
    target = round(pair_fraction * full)
    return max(0, min(full - base, target - base))


def allocate_voc_budget(
    *,
    key: str,
    seed: int,
    retrieved: list[list[tuple[int, float]]],
    policy: GatePolicy,
    pair_fraction: float,
    combined_scores: list[float],
    neural_scores: list[float],
    symbolic_scores: list[float],
) -> dict[str, list[PairRef]]:
    """Allocate exactly matched pair counts for learned and non-learned controls."""
    size = len(retrieved)
    if not all(len(scores) == size for scores in (combined_scores, neural_scores, symbolic_scores)):
        raise ValueError("policy scores must align with retrieved children")
    extras = _extras_by_child(retrieved, policy)
    budget = extra_pair_budget(retrieved, policy, pair_fraction)
    children = list(range(size))

    def ranked(values: list[float]) -> list[int]:
        return sorted(children, key=lambda child: (-float(values[child]), child))

    random_order = children.copy()
    random.Random(_stable_seed(seed, key)).shuffle(random_order)
    full = [ref for child in children for ref in extras[child]]
    # A small graph can have extra k=11..20 candidates while the prespecified
    # rounded 75%-of-full target still equals its k=10 prefix. The matched
    # expansion budget is then exactly zero, so every matched control must
    # receive the same empty intervention.
    uniform = [] if budget == 0 else _uniform_budget(extras, budget)
    return {
        "fixed_k10": [],
        "voc_combined": _take_budget(ranked(combined_scores), extras, budget),
        "voc_neural": _take_budget(ranked(neural_scores), extras, budget),
        "voc_symbolic": _take_budget(ranked(symbolic_scores), extras, budget),
        "random": _take_budget(random_order, extras, budget),
        "uniform_budget": uniform,
        "full_k20": full,
    }


def _set_f1(predicted: set[str], gold: set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    true_positive = len(predicted & gold)
    return 2.0 * true_positive / (len(predicted) + len(gold))


def _predicted_ancestors(node: str, parents: dict[str, str]) -> set[str]:
    result: set[str] = set()
    current = node
    while parents[current] != ROOT:
        current = parents[current]
        if current in result:
            raise ValueError("predicted parents must be acyclic")
        result.add(current)
    return result


def expansion_utility_targets(
    nodes: list[str],
    gold: set[tuple[str, str]],
    base: BranchingResult,
    full: BranchingResult,
    *,
    edge_weight: float,
    ancestor_weight: float,
) -> list[float]:
    """Counterfactual dev utility of full expansion for each child."""
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(gold)
    gold_parents: dict[str, set[str]] = {node: set() for node in nodes}
    for child, parent in gold:
        if child in gold_parents:
            gold_parents[child].add(parent)
    targets = []
    for node in nodes:
        gold_ancestors = set(nx.descendants(graph, node))
        base_ancestors = _predicted_ancestors(node, base.parents)
        full_ancestors = _predicted_ancestors(node, full.parents)
        ancestor_delta = _set_f1(full_ancestors, gold_ancestors) - _set_f1(base_ancestors, gold_ancestors)
        base_edge = (
            base.parents[node] in gold_parents[node]
            if gold_parents[node]
            else base.parents[node] == ROOT
        )
        full_edge = (
            full.parents[node] in gold_parents[node]
            if gold_parents[node]
            else full.parents[node] == ROOT
        )
        targets.append(
            ancestor_weight * ancestor_delta
            + edge_weight * (float(full_edge) - float(base_edge))
        )
    return targets
