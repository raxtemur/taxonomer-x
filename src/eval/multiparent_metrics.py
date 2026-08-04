"""Metrics and stage-wise attribution for multi-parent taxonomy induction."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from src.symbolic.multiparent_projection import DagProjectionResult


Edge = tuple[str, str]


def parents_by_child(edges: Iterable[Edge]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for child, parent in edges:
        grouped[child].add(parent)
    return dict(grouped)


def _prf(true_positive: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / gold if gold else 0.0
    return {
        "true_positive": true_positive,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / (predicted + gold) if predicted + gold else 0.0,
    }


def cardinality_detection_metrics(
    predicted_edges: Iterable[Edge], gold_edges: Iterable[Edge]
) -> dict[str, float | int]:
    """Child-level detection of whether a node has at least two parents."""
    predicted = parents_by_child(predicted_edges)
    gold = parents_by_child(gold_edges)
    predicted_multi = {child for child, parents in predicted.items() if len(parents) >= 2}
    gold_multi = {child for child, parents in gold.items() if len(parents) >= 2}
    return _prf(
        len(predicted_multi & gold_multi), len(predicted_multi), len(gold_multi)
    )


def multi_parent_set_metrics(
    predicted_edges: Iterable[Edge], gold_edges: Iterable[Edge]
) -> dict:
    """Evaluate both edges and complete sets on multi-parent decisions."""
    predicted = parents_by_child(predicted_edges)
    gold = parents_by_child(gold_edges)
    predicted_multi = {child for child, parents in predicted.items() if len(parents) >= 2}
    gold_multi = {child for child, parents in gold.items() if len(parents) >= 2}
    predicted_multi_edges = {
        (child, parent)
        for child in predicted_multi
        for parent in predicted.get(child, set())
    }
    gold_multi_edges = {
        (child, parent) for child in gold_multi for parent in gold.get(child, set())
    }
    edge_metrics = _prf(
        len(predicted_multi_edges & gold_multi_edges),
        len(predicted_multi_edges),
        len(gold_multi_edges),
    )
    union_children = predicted_multi | gold_multi
    exact_true_positive = sum(
        child in predicted_multi
        and child in gold_multi
        and predicted[child] == gold[child]
        for child in union_children
    )
    exact = _prf(exact_true_positive, len(predicted_multi), len(gold_multi))
    jaccards = []
    recovered_gold_parents = 0
    for child in gold_multi:
        predicted_parents = predicted.get(child, set())
        gold_parents = gold[child]
        union = predicted_parents | gold_parents
        jaccards.append(len(predicted_parents & gold_parents) / len(union))
        recovered_gold_parents += len(predicted_parents & gold_parents)
    return {
        "cardinality": cardinality_detection_metrics(predicted_edges, gold_edges),
        "multi_parent_edges": edge_metrics,
        "exact_parent_set": exact,
        "gold_child_mean_jaccard": (
            sum(jaccards) / len(jaccards) if jaccards else 0.0
        ),
        "gold_parent_recall": (
            recovered_gold_parents / len(gold_multi_edges)
            if gold_multi_edges
            else 0.0
        ),
    }


def candidate_edge_set(record, k: int) -> set[Edge]:
    candidates = set()
    for child_index, rows in enumerate(record.retrieved):
        child = record.nodes[child_index]
        for parent_index, _ in rows[:k]:
            candidates.add((child, record.nodes[parent_index]))
    return candidates


def retrieval_oracle(record, k: int) -> dict[str, float | int]:
    gold = parents_by_child(record.gold)
    candidate_edges = candidate_edge_set(record, k)
    gold_edges = set(record.gold)
    gold_multi = {child for child, parents in gold.items() if len(parents) >= 2}
    available_by_child = {
        child: {
            parent for parent in parents if (child, parent) in candidate_edges
        }
        for child, parents in gold.items()
    }
    all_available = sum(
        available_by_child[child] == gold[child] for child in gold_multi
    )
    at_least_one = sum(bool(available_by_child[child]) for child in gold_multi)
    available_edges = len(gold_edges & candidate_edges)
    return {
        "k": k,
        "gold_edges": len(gold_edges),
        "available_gold_edges": available_edges,
        "gold_edge_recall_ceiling": (
            available_edges / len(gold_edges) if gold_edges else 0.0
        ),
        "gold_multi_parent_children": len(gold_multi),
        "multi_parent_at_least_one_available": at_least_one,
        "multi_parent_all_parents_available": all_available,
        "multi_parent_at_least_one_rate": (
            at_least_one / len(gold_multi) if gold_multi else 0.0
        ),
        "multi_parent_complete_set_recall_ceiling": (
            all_available / len(gold_multi) if gold_multi else 0.0
        ),
    }


def conditional_ranking_audit(
    record,
    score_map: Mapping[tuple[int, int], float],
    k: int,
) -> dict[str, float | int]:
    """Audit second-parent ranking only where at least two gold parents exist."""
    gold = parents_by_child(record.gold)
    scored_children = 0
    both_available = 0
    both_in_top2 = 0
    second_gold_reciprocal_rank = 0.0
    for child_index, rows in enumerate(record.retrieved):
        child = record.nodes[child_index]
        if len(gold.get(child, ())) < 2:
            continue
        ranked = sorted(
            (
                (
                    record.nodes[parent_index],
                    float(score_map[(child_index, rank)]),
                )
                for rank, (parent_index, _) in enumerate(rows[:k])
                if (child_index, rank) in score_map
            ),
            key=lambda item: (-item[1], item[0]),
        )
        positions = [
            index + 1
            for index, (parent, _) in enumerate(ranked)
            if parent in gold[child]
        ]
        scored_children += 1
        if len(positions) >= 2:
            both_available += 1
            second_rank = sorted(positions)[1]
            second_gold_reciprocal_rank += 1.0 / second_rank
            both_in_top2 += int(second_rank <= 2)
    return {
        "gold_multi_parent_children": scored_children,
        "both_gold_available": both_available,
        "both_gold_top2": both_in_top2,
        "both_gold_top2_rate_conditional": (
            both_in_top2 / both_available if both_available else 0.0
        ),
        "second_gold_mrr_conditional": (
            second_gold_reciprocal_rank / both_available
            if both_available
            else 0.0
        ),
    }


def projection_stage_attribution(record, result: DagProjectionResult, k: int) -> dict:
    candidate_edges = candidate_edge_set(record, k)
    gold_edges = set(record.gold)
    local_edges = {
        (child, parent)
        for child, parents in result.local_parents.items()
        for parent in parents
    }
    projected_edges = set(result.taxonomy_edges)
    missing = gold_edges - candidate_edges
    available = gold_edges & candidate_edges
    return {
        "gold_edges": len(gold_edges),
        "retrieval_missing_gold_edges": len(missing),
        "retrieval_available_gold_edges": len(available),
        "local_recovered_available_gold_edges": len(local_edges & available),
        "projected_recovered_available_gold_edges": len(projected_edges & available),
        "local_false_positive_edges": len(local_edges - gold_edges),
        "projected_false_positive_edges": len(projected_edges - gold_edges),
        "projection_changed_children": len(result.changed_children),
        "projection_tension": result.regret,
        "local_parent_set": multi_parent_set_metrics(local_edges, gold_edges),
        "projected_parent_set": multi_parent_set_metrics(projected_edges, gold_edges),
    }
