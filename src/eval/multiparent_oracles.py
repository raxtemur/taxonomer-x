"""Evaluation-only oracle ablations for the multi-parent recovery funnel."""

from __future__ import annotations

from collections import defaultdict

from src.eval.multiparent_metrics import multi_parent_set_metrics
from src.symbolic.multiparent_projection import project_multiparent_dag


def _parents(edges):
    grouped = defaultdict(set)
    for child, parent in edges:
        grouped[child].add(parent)
    return dict(grouped)


def _candidate_parents(record, k: int) -> dict[str, set[str]]:
    result = {node: set() for node in record.nodes}
    for child_index, rows in enumerate(record.retrieved):
        child = record.nodes[child_index]
        for parent_index, _ in rows[:k]:
            parent = record.nodes[int(parent_index)]
            if parent != child:
                result[child].add(parent)
    return result


def complete_set_retrieval_oracle(record, k: int, max_parents: int = 2) -> dict:
    """Analytic exact-set ceiling before scoring or global projection."""
    gold = _parents(record.gold)
    candidates = _candidate_parents(record, k)
    gold_multi = {child: parents for child, parents in gold.items() if len(parents) >= 2}
    complete = {
        child for child, parents in gold_multi.items() if parents <= candidates[child]
    }
    structurally_eligible = {
        child for child, parents in gold_multi.items() if len(parents) <= max_parents
    }
    feasible_complete = complete & structurally_eligible
    oracle_edges = {
        (child, parent)
        for child in feasible_complete
        for parent in gold_multi[child]
    }
    parent_sets = multi_parent_set_metrics(oracle_edges, record.gold)
    total = len(gold_multi)
    return {
        "k": int(k),
        "gold_multi_parent_children": total,
        "complete_parent_sets_retrieved": len(complete),
        "structurally_eligible_gold_sets": len(structurally_eligible),
        "feasible_complete_parent_sets": len(feasible_complete),
        "complete_parent_set_recall": len(complete) / total if total else 0.0,
        "feasible_exact_recall_ceiling": len(feasible_complete) / total if total else 0.0,
        "parent_sets": parent_sets,
    }


def build_oracle_projections(
    record,
    transformed: dict,
    root_map: dict[str, float],
    *,
    penalty: float,
    score_k: int,
    multi_parent_cap: int,
    solver_name: str,
    time_limit: float,
) -> tuple[dict, dict]:
    """Run isolated first-parent, cardinality, and in-pool pair oracles."""
    gold = _parents(record.gold)
    gold_multi = {child: parents for child, parents in gold.items() if len(parents) >= 2}
    candidates = _candidate_parents(record, score_k)
    edge_scores = {
        (candidate.child, candidate.parent): float(candidate.score)
        for candidate in transformed["candidates"]
    }

    first_edges = set()
    for child, parents in gold_multi.items():
        available = parents & candidates.get(child, set())
        if available:
            parent = max(
                available,
                key=lambda candidate: (
                    edge_scores.get((child, candidate), float("-inf")),
                    candidate,
                ),
            )
            first_edges.add((child, parent))
    if any(not isinstance(edge, tuple) or len(edge) != 2 for edge in first_edges):
        raise AssertionError("malformed oracle first-parent edge")

    common = dict(
        nodes=record.nodes,
        candidates=transformed["candidates"],
        root_scores=root_map,
        max_parents=2,
        second_parent_threshold=float(penalty),
        cardinality_scores=transformed["cardinality_scores"],
        pair_scores=transformed["pair_scores"],
        solver_name=solver_name,
        time_limit=time_limit,
    )
    first_parent = project_multiparent_dag(
        **common,
        multi_parent_cap=multi_parent_cap,
        required_edges=first_edges,
    )

    cardinality_required = {
        child for child in gold_multi if len(candidates.get(child, set())) >= 2
    }
    cardinality_forbidden = set(record.nodes) - cardinality_required
    oracle_cardinality = project_multiparent_dag(
        **common,
        multi_parent_cap=len(cardinality_required),
        required_multi_parent_children=cardinality_required,
        forbidden_multi_parent_children=cardinality_forbidden,
    )

    pair_required_children = {
        child
        for child, parents in gold_multi.items()
        if len(parents) == 2 and parents <= candidates.get(child, set())
    }
    pair_required_edges = {
        (child, parent)
        for child in pair_required_children
        for parent in gold_multi[child]
    }
    pair_forbidden_children = set(record.nodes) - pair_required_children
    oracle_pair = project_multiparent_dag(
        **common,
        multi_parent_cap=len(pair_required_children),
        required_edges=pair_required_edges,
        required_multi_parent_children=pair_required_children,
        forbidden_multi_parent_children=pair_forbidden_children,
    )
    audits = {
        "oracle_first_parent": {
            "required_gold_first_parent_edges": len(first_edges),
            "gold_multi_parent_children": len(gold_multi),
        },
        "oracle_cardinality": {
            "required_multi_parent_children": len(cardinality_required),
            "unavailable_cardinality_children": len(gold_multi) - len(cardinality_required),
        },
        "oracle_pair_selection_within_retrieved_pool": {
            "required_exact_pairs": len(pair_required_children),
            "required_gold_edges": len(pair_required_edges),
        },
    }
    return {
        "oracle_first_parent": first_parent,
        "oracle_cardinality": oracle_cardinality,
        "oracle_pair_selection_within_retrieved_pool": oracle_pair,
    }, audits
