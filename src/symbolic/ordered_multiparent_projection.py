"""Linear-time exact decoder for an acyclic-by-construction candidate graph."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import networkx as nx

from src.symbolic.multiparent_projection import (
    CardinalityScores,
    DagCandidate,
    DagProjectionResult,
    PairScores,
    _choice_value,
    _deduplicate_candidates,
)


def project_ordered_multiparent_dag(
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
    topological_order: Mapping[str, float],
    root_scores: dict[str, float] | None = None,
    *,
    max_parents: int = 2,
    second_parent_threshold: float = 0.0,
    multi_parent_cap: int | None = None,
    cardinality_scores: CardinalityScores | None = None,
    pair_scores: PairScores | None = None,
) -> DagProjectionResult:
    """Decode exactly when every candidate points forward in a fixed order.

    Once the candidate supergraph is a DAG, child decisions are independent
    except for the global number of second-parent slots. The exact solution is
    obtained by selecting each child's best zero/one-parent option and then the
    largest positive two-parent marginal gains up to the budget.
    """
    vocabulary = set(nodes)
    if not vocabulary:
        raise ValueError("nodes must not be empty")
    if set(topological_order) != vocabulary:
        missing = vocabulary - set(topological_order)
        extra = set(topological_order) - vocabulary
        raise ValueError(f"topological order mismatch: missing={missing}, extra={extra}")
    if max_parents not in (1, 2):
        raise ValueError("max_parents must be one or two")
    if multi_parent_cap is not None and multi_parent_cap < 0:
        raise ValueError("multi_parent_cap must be non-negative")

    ordered_nodes = sorted(vocabulary)
    roots = {node: float((root_scores or {}).get(node, 0.0)) for node in ordered_nodes}
    cardinalities = cardinality_scores or {}
    pairs = pair_scores or {}
    edges = _deduplicate_candidates(vocabulary, candidates)
    invalid = [
        (edge.child, edge.parent)
        for edge in edges
        if not topological_order[edge.parent] > topological_order[edge.child]
    ]
    if invalid:
        raise ValueError(
            f"candidate supergraph violates the declared order: {invalid[:5]!r}"
        )
    by_child: dict[str, list[DagCandidate]] = {node: [] for node in ordered_nodes}
    for edge in edges:
        by_child[edge.child].append(edge)
    edge_scores = {(edge.child, edge.parent): edge.score for edge in edges}

    best_base: dict[str, tuple[tuple[str, ...], float]] = {}
    best_pair: dict[str, tuple[tuple[str, ...], float] | None] = {}
    for child in ordered_nodes:
        rows = sorted(by_child[child], key=lambda row: row.parent)
        base_options = [()]
        base_options.extend((row.parent,) for row in rows)
        base_scored = [
            (
                parents,
                _choice_value(
                    child,
                    parents,
                    edge_scores,
                    roots,
                    cardinalities,
                    pairs,
                    second_parent_threshold,
                ),
            )
            for parents in base_options
        ]
        best_base[child] = min(
            base_scored, key=lambda item: (-item[1], len(item[0]), item[0])
        )
        pair_scored = [
            (
                tuple(sorted((left.parent, right.parent))),
                _choice_value(
                    child,
                    tuple(sorted((left.parent, right.parent))),
                    edge_scores,
                    roots,
                    cardinalities,
                    pairs,
                    second_parent_threshold,
                ),
            )
            for left, right in combinations(rows, 2)
        ]
        best_pair[child] = (
            min(pair_scored, key=lambda item: (-item[1], item[0]))
            if max_parents == 2 and pair_scored
            else None
        )

    gains = []
    for child in ordered_nodes:
        if best_pair[child] is not None:
            gain = best_pair[child][1] - best_base[child][1]
            gains.append((gain, child))
    gains.sort(key=lambda item: (-item[0], item[1]))
    cap = (
        len(ordered_nodes)
        if multi_parent_cap is None
        else min(int(multi_parent_cap), len(ordered_nodes))
    )
    selected_multi = {
        child for gain, child in gains[:cap] if gain > 1e-12
    }
    parents = {
        child: (
            best_pair[child][0]
            if child in selected_multi and best_pair[child] is not None
            else best_base[child][0]
        )
        for child in ordered_nodes
    }
    selected_values = {
        child: (
            best_pair[child][1]
            if child in selected_multi and best_pair[child] is not None
            else best_base[child][1]
        )
        for child in ordered_nodes
    }
    local_parents = {
        child: (
            best_pair[child][0]
            if best_pair[child] is not None
            and best_pair[child][1] > best_base[child][1] + 1e-12
            else best_base[child][0]
        )
        for child in ordered_nodes
    }
    local_values = {
        child: max(
            best_base[child][1],
            best_pair[child][1] if best_pair[child] is not None else float("-inf"),
        )
        for child in ordered_nodes
    }
    sacrifices = {
        child: max(0.0, local_values[child] - selected_values[child])
        for child in ordered_nodes
    }
    selected_score = sum(selected_values.values())
    local_score = sum(local_values.values())

    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_nodes)
    graph.add_edges_from(
        (child, parent)
        for child, parent_values in parents.items()
        for parent in parent_values
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise AssertionError("ordered candidate decoder produced a cycle")
    roots_selected = tuple(sorted(child for child, values in parents.items() if not values))
    multi_children = tuple(
        sorted(child for child, values in parents.items() if len(values) == 2)
    )
    return DagProjectionResult(
        parents=parents,
        local_parents=local_parents,
        score=selected_score,
        local_score=local_score,
        regret=max(0.0, local_score - selected_score),
        graph=graph,
        changed_children=tuple(
            sorted(child for child in ordered_nodes if parents[child] != local_parents[child])
        ),
        score_sacrifice=sacrifices,
        roots=roots_selected,
        solver_status="Optimal",
        solver_name="ordered",
        solve_seconds=0.0,
        max_parents=max_parents,
        second_parent_threshold=second_parent_threshold,
        multi_parent_cap=None if multi_parent_cap is None else cap,
        unpenalized_score=selected_score
        + second_parent_threshold * len(multi_children),
        extra_parent_count=len(multi_children),
    )
