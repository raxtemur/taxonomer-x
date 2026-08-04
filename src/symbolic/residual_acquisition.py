"""Residual neuro-symbolic evidence acquisition with exact-budget controls.

The neural gate predicts development-set value of additional candidates.  A
second gate is trained only on out-of-fold neural residuals and structural
features.  This makes the scientific question explicit: does structure carry
information not already available in neural uncertainty?
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass

import networkx as nx

from src.symbolic.budgeted_expansion import GatePolicy, PairRef
from src.symbolic.maximum_branching import BranchCandidate, BranchingResult, ROOT, maximum_branching
from src.symbolic.value_of_computation import (
    NEURAL_FEATURES,
    SYMBOLIC_FEATURES,
    RidgeGate,
    fit_ridge_gate,
    local_gate_features,
    predict_gate,
)


SENSITIVITY_FEATURES = (
    "projection_edited",
    "root_conflict",
    "score_sacrifice",
    "exchange_uncertainty",
    "subtree_size_log",
    "parent_load_log",
)
RESIDUAL_SYMBOLIC_FEATURES = SYMBOLIC_FEATURES + SENSITIVITY_FEATURES


@dataclass(frozen=True)
class ResidualGate:
    neural: RidgeGate
    symbolic_residual: RidgeGate
    folds: int

    def as_dict(self) -> dict:
        return {
            "neural": self.neural.as_dict(),
            "symbolic_residual": self.symbolic_residual.as_dict(),
            "folds": self.folds,
            "residual_target": "utility_minus_out_of_fold_neural_prediction",
        }


def _choices(
    child: int,
    nodes: list[str],
    retrieved: list[list[tuple[int, float]]],
    scores: dict[PairRef, float],
    root_scores: list[float],
    policy: GatePolicy,
) -> list[tuple[str, float]]:
    result = [(ROOT, float(root_scores[child]))]
    result.extend(
        (nodes[retrieved[child][rank][0]], float(scores[(child, rank)]))
        for rank in range(min(policy.base_k, len(retrieved[child])))
    )
    # Retrieval caches should not contain duplicates, but keeping the maximum
    # makes the sensitivity audit total under malformed external caches.
    best: dict[str, float] = {}
    for parent, score in result:
        best[parent] = max(score, best.get(parent, -math.inf))
    return sorted(best.items())


def branching_sensitivity_features(
    nodes: list[str],
    retrieved: list[list[tuple[int, float]]],
    scores: dict[PairRef, float],
    root_scores: list[float],
    policy: GatePolicy,
) -> tuple[list[dict[str, float]], BranchingResult, dict]:
    """Add label-free features from one exact base-prefix projection.

    ``exchange_uncertainty`` is based on the best *single-child* feasible
    alternative with all other selected parents fixed.  It is a local
    sensitivity certificate, not a claim about full Edmonds reduced costs.
    """
    rows = local_gate_features(nodes, retrieved, scores, root_scores, policy)
    candidates = []
    for child in range(len(nodes)):
        for parent, score in _choices(child, nodes, retrieved, scores, root_scores, policy):
            if parent != ROOT:
                candidates.append(BranchCandidate(nodes[child], parent, score))
    base = maximum_branching(
        nodes, candidates, {node: float(root_scores[i]) for i, node in enumerate(nodes)}
    )

    selected_scores: dict[str, float] = {}
    local_parents: dict[str, str] = {}
    local_scores: dict[str, float] = {}
    for child, node in enumerate(nodes):
        options = _choices(child, nodes, retrieved, scores, root_scores, policy)
        local_parent, local_score = max(options, key=lambda item: (item[1], item[0]))
        local_parents[node], local_scores[node] = local_parent, local_score
        selected_scores[node] = dict(options)[base.parents[node]]

    graph = base.graph  # public orientation child -> parent
    parent_load: dict[str, int] = {node: 0 for node in nodes}
    for parent in base.parents.values():
        if parent != ROOT:
            parent_load[parent] += 1
    exchange_margins = []
    for child, node in enumerate(nodes):
        selected_parent = base.parents[node]
        work = graph.copy()
        if selected_parent != ROOT and work.has_edge(node, selected_parent):
            work.remove_edge(node, selected_parent)
        feasible_alternatives = []
        for parent, score in _choices(child, nodes, retrieved, scores, root_scores, policy):
            if parent == selected_parent:
                continue
            if parent == ROOT or not nx.has_path(work, parent, node):
                feasible_alternatives.append(score)
        alternative = max(feasible_alternatives, default=selected_scores[node])
        margin = max(0.0, selected_scores[node] - alternative)
        exchange_margins.append(margin)
        subtree_size = len(nx.ancestors(graph, node)) + 1
        edited = float(local_parents[node] != selected_parent)
        rows[child].update({
            "projection_edited": edited,
            "root_conflict": float((local_parents[node] == ROOT) != (selected_parent == ROOT)),
            "score_sacrifice": max(0.0, local_scores[node] - selected_scores[node]),
            "exchange_uncertainty": 1.0 / (1.0 + margin),
            "subtree_size_log": math.log1p(subtree_size),
            "parent_load_log": math.log1p(parent_load.get(selected_parent, 0)),
        })
    if not all(math.isfinite(float(row[name])) for row in rows for name in RESIDUAL_SYMBOLIC_FEATURES):
        raise ValueError("all residual-acquisition features must be finite")
    audit = {
        "base_score": base.score,
        "projection_edits": sum(row["projection_edited"] for row in rows),
        "root_conflicts": sum(row["root_conflict"] for row in rows),
        "constraint_tension": sum(row["score_sacrifice"] for row in rows),
        "minimum_exchange_margin": min(exchange_margins, default=0.0),
        "feature_semantics": "single-child feasible exchange with other parents fixed",
    }
    return rows, base, audit


def fit_residual_gate(
    graph_rows: list[tuple[str, list[dict[str, float]], list[float]]],
    *,
    l2: float,
    folds: int,
) -> tuple[ResidualGate, dict]:
    """Fit graph-folded neural predictions, then their structural residual."""
    if folds < 2 or len(graph_rows) < folds:
        raise ValueError("residual fitting needs at least two graph folds")
    oof_neural: list[float] = []
    oof_targets: list[float] = []
    oof_features: list[dict[str, float]] = []
    for fold in range(folds):
        train = [row for index, row in enumerate(graph_rows) if index % folds != fold]
        heldout = [row for index, row in enumerate(graph_rows) if index % folds == fold]
        train_x = [item for _, features, _ in train for item in features]
        train_y = [item for _, _, targets in train for item in targets]
        gate = fit_ridge_gate(train_x, train_y, feature_names=NEURAL_FEATURES, l2=l2)
        for _, features, targets in heldout:
            oof_features.extend(features)
            oof_targets.extend(targets)
            oof_neural.extend(predict_gate(gate, features))
    residuals = [target - prediction for target, prediction in zip(oof_targets, oof_neural)]
    residual_gate = fit_ridge_gate(
        oof_features, residuals, feature_names=RESIDUAL_SYMBOLIC_FEATURES, l2=l2
    )
    all_x = [item for _, features, _ in graph_rows for item in features]
    all_y = [item for _, _, targets in graph_rows for item in targets]
    neural_gate = fit_ridge_gate(all_x, all_y, feature_names=NEURAL_FEATURES, l2=l2)
    neural_mean = sum(oof_neural) / len(oof_neural)
    target_mean = sum(oof_targets) / len(oof_targets)
    total = sum((value - target_mean) ** 2 for value in oof_targets)
    residual_error = sum((target - pred) ** 2 for target, pred in zip(oof_targets, oof_neural))
    audit = {
        "fold_unit": "node_disjoint_development_graph",
        "folds": folds,
        "n_graphs": len(graph_rows),
        "n_nodes": len(oof_targets),
        "oof_neural_mean": neural_mean,
        "residual_mean": sum(residuals) / len(residuals),
        "oof_neural_r2": 1.0 - residual_error / total if total > 0 else 0.0,
        "test_labels_used": False,
    }
    return ResidualGate(neural_gate, residual_gate, folds), audit


def residual_policy_scores(
    gate: ResidualGate,
    rows: list[dict[str, float]],
    *,
    seed: int,
    key: str,
    deciles: int = 10,
) -> dict[str, list[float]]:
    neural = predict_gate(gate.neural, rows)
    symbolic = predict_gate(gate.symbolic_residual, rows)
    combined = [n + s for n, s in zip(neural, symbolic)]
    rng = random.Random(_stable_seed(seed, key, "global"))
    global_permuted = symbolic.copy()
    rng.shuffle(global_permuted)

    within = symbolic.copy()
    order = sorted(range(len(neural)), key=lambda index: (neural[index], index))
    for bin_index in range(deciles):
        members = order[bin_index * len(order) // deciles : (bin_index + 1) * len(order) // deciles]
        values = [within[index] for index in members]
        random.Random(_stable_seed(seed, key, f"decile:{bin_index}")).shuffle(values)
        for index, value in zip(members, values):
            within[index] = value
    return {
        "neural_only": neural,
        "residual_neurosymbolic": combined,
        "symbolic_residual_only": symbolic,
        "global_permuted_symbolic": [n + s for n, s in zip(neural, global_permuted)],
        "decile_permuted_symbolic": [n + s for n, s in zip(neural, within)],
    }


def allocate_exact_budget(
    retrieved: list[list[tuple[int, float]]],
    policy: GatePolicy,
    pair_fraction: float,
    scores: dict[str, list[float]],
    *,
    seed: int,
    key: str,
) -> dict[str, list[PairRef]]:
    """Allocate identical per-graph pair counts for all ranked controls."""
    base = sum(min(policy.base_k, len(row)) for row in retrieved)
    full = sum(min(policy.max_k, len(row)) for row in retrieved)
    budget = max(0, min(full - base, round(pair_fraction * full) - base))
    extras = {
        child: [(child, rank) for rank in range(policy.base_k, min(policy.max_k, len(row)))]
        for child, row in enumerate(retrieved)
    }

    def take(order: list[int]) -> list[PairRef]:
        result = []
        for child in order:
            for ref in extras[child]:
                if len(result) == budget:
                    return result
                result.append(ref)
        return result

    children = list(range(len(retrieved)))
    allocations = {
        name: take(sorted(children, key=lambda child: (-float(values[child]), child)))
        for name, values in scores.items()
    }
    shuffled = children.copy()
    random.Random(_stable_seed(seed, key, "random-control")).shuffle(shuffled)
    allocations["random"] = take(shuffled)
    # Round-robin is independent of scores and spreads each added rank evenly.
    uniform_order = [(child, rank) for rank in range(policy.base_k, policy.max_k)
                     for child in children if rank < len(retrieved[child])]
    allocations["uniform_budget"] = uniform_order[:budget]
    allocations["fixed_k10"] = []
    allocations["full_k20"] = [ref for child in children for ref in extras[child]]
    matched = {len(value) for name, value in allocations.items() if name not in {"fixed_k10", "full_k20"}}
    if matched != {budget}:
        raise AssertionError(f"equal-budget allocation failed: {matched} != {{{budget}}}")
    return allocations


def _stable_seed(seed: int, key: str, namespace: str) -> int:
    digest = hashlib.sha256(f"residual:{seed}:{key}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
