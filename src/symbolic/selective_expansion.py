"""Tension-gated candidate expansion followed by exact symbolic projection."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch

from src.symbolic.maximum_branching import BranchCandidate, ROOT, maximum_branching


@dataclass(frozen=True)
class ExpansionPolicy:
    base_k: int = 20
    max_k: int = 50
    root_margin: float = 0.25
    edge_gap: float = 0.25

    def __post_init__(self) -> None:
        if self.base_k < 1 or self.max_k < self.base_k:
            raise ValueError("require 1 <= base_k <= max_k")
        if self.root_margin < 0 or self.edge_gap < 0:
            raise ValueError("margins must be non-negative")


def _selected_indices(child_index: torch.Tensor, policy: ExpansionPolicy, expanded: set[int]) -> list[int]:
    counts: dict[int, int] = {}
    selected = []
    for index, child in enumerate(child_index.tolist()):
        rank = counts.get(child, 0)
        counts[child] = rank + 1
        if rank < (policy.max_k if child in expanded else policy.base_k):
            selected.append(index)
    return selected


def _decode(
    nodes: list[str],
    child_index: torch.Tensor,
    parent_index: torch.Tensor,
    edge_scores: torch.Tensor,
    root_scores: torch.Tensor,
    selected: list[int],
):
    candidates = [
        BranchCandidate(nodes[int(child_index[i])], nodes[int(parent_index[i])], float(edge_scores[i]))
        for i in selected
    ]
    return maximum_branching(
        nodes,
        candidates,
        {node: float(root_scores[index]) for index, node in enumerate(nodes)},
    )


def selective_maximum_branching(
    nodes: list[str],
    child_index: torch.Tensor,
    parent_index: torch.Tensor,
    edge_scores: torch.Tensor,
    root_scores: torch.Tensor,
    policy: ExpansionPolicy,
) -> dict:
    """Decode K_base, inspect structural tension, expand only triggered nodes."""
    child_index = child_index.long().cpu()
    parent_index = parent_index.long().cpu()
    edge_scores = edge_scores.float().cpu()
    root_scores = root_scores.float().cpu()
    base_selected = _selected_indices(child_index, policy, set())
    base = _decode(nodes, child_index, parent_index, edge_scores, root_scores, base_selected)

    by_child: dict[int, list[int]] = {index: [] for index in range(len(nodes))}
    for edge_index in base_selected:
        by_child[int(child_index[edge_index])].append(edge_index)
    local_parents: dict[str, str] = {}
    triggers: dict[int, set[str]] = {}
    for child, indices in by_child.items():
        ordered = sorted(indices, key=lambda i: float(edge_scores[i]), reverse=True)
        best_edge = float(edge_scores[ordered[0]]) if ordered else -float("inf")
        if not ordered or float(root_scores[child]) >= best_edge:
            local_parents[nodes[child]] = ROOT
        else:
            local_parents[nodes[child]] = nodes[int(parent_index[ordered[0]])]
        if ordered and abs(float(root_scores[child]) - best_edge) <= policy.root_margin:
            triggers.setdefault(child, set()).add("root_edge_margin")
        if len(ordered) > 1 and best_edge - float(edge_scores[ordered[1]]) <= policy.edge_gap:
            triggers.setdefault(child, set()).add("top1_top2_gap")

    local_graph = nx.DiGraph()
    local_graph.add_nodes_from(nodes)
    local_graph.add_edges_from((child, parent) for child, parent in local_parents.items() if parent != ROOT)
    for component in nx.strongly_connected_components(local_graph):
        if len(component) > 1:
            for node in component:
                triggers.setdefault(nodes.index(node), set()).add("local_cycle")

    for child, parent in base.parents.items():
        if parent != local_parents[child]:
            triggers.setdefault(nodes.index(child), set()).add("exact_edit")

    expanded = set(triggers)
    selected = _selected_indices(child_index, policy, expanded)
    final = _decode(nodes, child_index, parent_index, edge_scores, root_scores, selected)
    return {
        "result": final,
        "base_result": base,
        "expanded_indices": sorted(expanded),
        "triggers": {nodes[index]: sorted(reasons) for index, reasons in sorted(triggers.items())},
        "base_pairs": len(base_selected),
        "scored_pairs": len(selected),
        "full_pairs": len(_selected_indices(child_index, policy, set(range(len(nodes))))),
        "activation_rate": len(expanded) / max(1, len(nodes)),
        "pair_fraction": len(selected) / max(1, len(_selected_indices(child_index, policy, set(range(len(nodes)))))),
        "guarantees": {"one_parent_or_root": True, "acyclic": True, "exact_final_projection": True},
    }
