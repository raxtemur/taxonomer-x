"""Loss-augmented maximum-branching hinge with an exact discrete oracle."""

from __future__ import annotations

import torch

from src.symbolic.maximum_branching import BranchCandidate, ROOT, maximum_branching


def structured_hinge(
    nodes: list[str],
    edge_pairs: list[tuple[str, str]],
    edge_scores: torch.Tensor,
    root_scores: torch.Tensor,
    gold_parents: dict[str, str],
    margin: float = 1.0,
    node_costs: dict[str, float] | None = None,
) -> torch.Tensor:
    """Compute a subgradient-compatible structured hinge for one graph.

    Loss-augmented inference is discrete and runs on detached scores. The
    selected and gold structures are then indexed from the original tensors,
    giving the standard structured-SVM subgradient.
    """
    if edge_scores.ndim != 1 or len(edge_pairs) != edge_scores.numel():
        raise ValueError("edge_pairs and edge_scores must be aligned 1-D values")
    if root_scores.numel() != len(nodes):
        raise ValueError("root_scores must contain one value per node")
    node_index = {node: index for index, node in enumerate(nodes)}
    edge_index = {pair: index for index, pair in enumerate(edge_pairs)}
    augmented: list[BranchCandidate] = []
    node_costs = node_costs or {}
    for index, (child, parent) in enumerate(edge_pairs):
        weight = float(node_costs.get(child, 1.0))
        cost = margin * weight if child in gold_parents and gold_parents[child] != parent else 0.0
        augmented.append(BranchCandidate(child, parent, float(edge_scores[index].detach()) + cost))
    augmented_roots = {
        child: float(root_scores[index].detach()) + (margin * float(node_costs.get(child, 1.0)) if child in gold_parents and gold_parents[child] != ROOT else 0.0)
        for index, child in enumerate(nodes)
    }
    predicted = maximum_branching(nodes, augmented, augmented_roots).parents

    predicted_score = edge_scores.new_zeros(())
    gold_score = edge_scores.new_zeros(())
    delta = 0.0
    for child in nodes:
        predicted_parent = predicted[child]
        if child not in gold_parents:
            continue
        gold_parent = gold_parents[child]
        if predicted_parent == ROOT:
            predicted_score = predicted_score + root_scores[node_index[child]]
        else:
            predicted_score = predicted_score + edge_scores[edge_index[(child, predicted_parent)]]
        if gold_parent == ROOT:
            gold_score = gold_score + root_scores[node_index[child]]
        else:
            if (child, gold_parent) not in edge_index:
                raise ValueError(f"Gold edge {(child, gold_parent)!r} is absent from candidates")
            gold_score = gold_score + edge_scores[edge_index[(child, gold_parent)]]
        delta += margin * float(node_costs.get(child, 1.0)) * (predicted_parent != gold_parent)
    return torch.relu(predicted_score - gold_score + edge_scores.new_tensor(delta))
