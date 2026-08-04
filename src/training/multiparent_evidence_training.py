"""Memory-bounded fitting for frozen multi-parent evidence models."""

from __future__ import annotations

from itertools import combinations
from typing import Mapping

from src.eval.score_calibration import calibration_metrics, fit_logistic_calibration
from src.models.multiparent_evidence import (
    FrozenMultiParentEvidenceModel,
    PairRef,
    _cardinality_features,
    _pair_features,
    _ranked_rows,
)


def fit_sampled_frozen_multiparent_evidence(
    records,
    score_maps: list[Mapping[PairRef, float]],
    *,
    max_k: int,
    negative_pairs_per_child: int = 8,
    l2: float = 1.0,
) -> tuple[FrozenMultiParentEvidenceModel, dict]:
    """Fit all heads while retaining every positive and bounded negatives."""
    if len(records) != len(score_maps):
        raise ValueError("records and score maps must align")
    if negative_pairs_per_child < 1:
        raise ValueError("negative_pairs_per_child must be positive")

    edge_features = []
    edge_labels = []
    indexed_rows = []
    for record, score_map in zip(records, score_maps):
        gold = set(record.gold)
        for child in range(len(record.nodes)):
            rows = _ranked_rows(record, score_map, child, max_k)
            indexed_rows.append((record, child, rows))
            for row in rows:
                edge_features.append([row["raw_score"]])
                edge_labels.append(
                    int((record.nodes[child], row["parent"]) in gold)
                )
    edge = fit_logistic_calibration(edge_features, edge_labels, l2=l2)

    cardinality_features = []
    cardinality_labels = []
    pair_features = []
    pair_labels = []
    for record, child, rows in indexed_rows:
        gold_parents = {
            parent
            for gold_child, parent in record.gold
            if gold_child == record.nodes[child]
        }
        logits = edge.log_odds([[row["raw_score"]] for row in rows]).tolist() if rows else []
        cardinality_features.append(_cardinality_features(logits))
        cardinality_labels.append(int(len(gold_parents) >= 2))
        positives = []
        negatives = []
        for (left, left_logit), (right, right_logit) in combinations(
            zip(rows, logits), 2
        ):
            item = (
                _pair_features(left, right, left_logit, right_logit),
                int({left["parent"], right["parent"]}.issubset(gold_parents)),
            )
            (positives if item[1] else negatives).append(item)
        selected = positives + negatives[:negative_pairs_per_child]
        pair_features.extend(features for features, _ in selected)
        pair_labels.extend(label for _, label in selected)

    cardinality = fit_logistic_calibration(
        cardinality_features, cardinality_labels, l2=l2
    )
    pair = (
        fit_logistic_calibration(pair_features, pair_labels, l2=l2)
        if pair_labels and 0 < sum(pair_labels) < len(pair_labels)
        else None
    )
    model = FrozenMultiParentEvidenceModel(
        edge=edge,
        cardinality=cardinality,
        pair=pair,
        max_k=max_k,
    )
    diagnostics = {
        "edge": calibration_metrics(edge_labels, edge.probabilities(edge_features)),
        "cardinality": calibration_metrics(
            cardinality_labels,
            cardinality.probabilities(cardinality_features),
        ),
        "pair": (
            calibration_metrics(pair_labels, pair.probabilities(pair_features))
            if pair is not None
            else None
        ),
        "edge_examples": len(edge_labels),
        "cardinality_examples": len(cardinality_labels),
        "pair_examples": len(pair_labels),
        "pair_positive_examples": sum(pair_labels),
        "negative_pairs_per_child": negative_pairs_per_child,
    }
    return model, diagnostics
