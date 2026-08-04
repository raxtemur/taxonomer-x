"""Interpretable scalar features and ablations for candidate-graph context."""

from __future__ import annotations

from collections import Counter, defaultdict

import networkx as nx
import torch


FEATURE_NAMES = (
    "parent_popularity",
    "sibling_coherence",
    "parent_path_score",
    "parent_depth",
    "parent_centrality",
    "candidate_entropy",
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def _top1_depths(children: list[int], parents: list[int], scores: torch.Tensor) -> dict[int, float]:
    best: dict[int, tuple[int, float]] = {}
    for child, parent, score in zip(children, parents, scores.tolist()):
        if child not in best or score > best[child][1]:
            best[child] = (parent, score)
    depths: dict[int, float] = {}
    all_nodes = set(children) | set(parents)
    for start in all_nodes:
        current, seen, depth = start, {start}, 0
        while current in best:
            nxt = best[current][0]
            if nxt in seen:
                depth = 0  # cyclic local hypotheses have no trusted depth
                break
            seen.add(nxt)
            current = nxt
            depth += 1
        depths[start] = float(depth)
    return depths


def candidate_graph_features(
    child_index: torch.Tensor,
    parent_index: torch.Tensor,
    local_scores: torch.Tensor,
) -> torch.Tensor:
    """Return six z-normalized features aligned with candidate edges."""
    if local_scores.ndim != 1 or local_scores.numel() != child_index.numel():
        raise ValueError("indices and local_scores must be aligned 1-D tensors")
    if parent_index.numel() != child_index.numel():
        raise ValueError("child_index and parent_index must be aligned")
    if local_scores.numel() == 0:
        return local_scores.new_zeros((0, len(FEATURE_NAMES)))
    children, parents = child_index.tolist(), parent_index.tolist()
    scores = local_scores.detach().float().cpu()
    popularity = Counter(parents)
    by_child: dict[int, list[int]] = defaultdict(list)
    by_parent: dict[int, list[int]] = defaultdict(list)
    graph = nx.DiGraph()
    for edge, (child, parent) in enumerate(zip(children, parents)):
        by_child[child].append(edge)
        by_parent[parent].append(edge)
        weight = float(torch.sigmoid(scores[edge]))
        if not graph.has_edge(child, parent) or weight > graph[child][parent]["weight"]:
            graph.add_edge(child, parent, weight=weight)
    centrality = nx.pagerank(graph, alpha=0.85, weight="weight")
    depths = _top1_depths(children, parents, scores)
    entropy: dict[int, float] = {}
    for child, edges in by_child.items():
        probabilities = scores[edges].softmax(0)
        entropy[child] = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    features = []
    for child, parent in zip(children, parents):
        sibling_edges = by_parent[parent]
        sibling_coherence = float(torch.sigmoid(scores[sibling_edges]).mean())
        parent_paths = by_child.get(parent, [])
        parent_path = float(scores[parent_paths].max()) if parent_paths else 0.0
        features.append((
            float(torch.log1p(torch.tensor(popularity[parent]))),
            sibling_coherence,
            parent_path,
            float(torch.log1p(torch.tensor(depths.get(parent, 0.0)))),
            float(centrality.get(parent, 0.0)),
            entropy[child],
        ))
    result = torch.tensor(features, dtype=local_scores.dtype, device=local_scores.device)
    return (result - result.mean(0, keepdim=True)) / result.std(0, keepdim=True, unbiased=False).clamp_min(1e-6)


def ablate_features(features: torch.Tensor, disabled: set[str] | list[str] | tuple[str, ...]) -> torch.Tensor:
    """Zero selected normalized feature columns for controlled ablations."""
    unknown = set(disabled) - set(FEATURE_INDEX)
    if unknown:
        raise ValueError(f"Unknown graph features: {sorted(unknown)}")
    result = features.clone()
    for name in disabled:
        result[..., FEATURE_INDEX[name]] = 0
    return result
