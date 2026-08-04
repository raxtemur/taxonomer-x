"""Learned cardinality and conditional parent-pair evidence from frozen scores."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping

import numpy as np

from src.eval.score_calibration import (
    LogisticCalibration,
    calibration_metrics,
    fit_logistic_calibration,
)
from src.symbolic.multiparent_projection import DagCandidate, PairKey


PairRef = tuple[int, int]


def _ranked_rows(record, score_map: Mapping[PairRef, float], child: int, k: int):
    rows = []
    for rank, (parent, retrieval_score) in enumerate(record.retrieved[child][:k]):
        if int(parent) == child or not np.isfinite(retrieval_score):
            continue
        reference = (child, rank)
        if reference in score_map:
            rows.append(
                {
                    "rank": rank,
                    "parent_index": int(parent),
                    "parent": record.nodes[parent],
                    "raw_score": float(score_map[reference]),
                    "retrieval_score": float(retrieval_score),
                }
            )
    return rows


def _cardinality_features(edge_logits: list[float]) -> list[float]:
    ranked = sorted(edge_logits, reverse=True)
    top1 = ranked[0] if ranked else 0.0
    top2 = ranked[1] if len(ranked) > 1 else top1
    values = np.asarray(ranked or [0.0], dtype=np.float64)
    maximum = float(values.max())
    logsumexp = maximum + float(np.log(np.exp(values - maximum).sum()))
    return [
        top1,
        top2,
        top1 - top2,
        float(values.mean()),
        float(values.std()),
        logsumexp,
        float(len(ranked)),
    ]


def _pair_features(left: dict, right: dict, left_logit: float, right_logit: float):
    low, high = sorted((left_logit, right_logit))
    return [
        low,
        high,
        left_logit + right_logit,
        abs(left_logit - right_logit),
        float(min(left["rank"], right["rank"])),
        float(max(left["rank"], right["rank"])),
        min(left["retrieval_score"], right["retrieval_score"]),
        max(left["retrieval_score"], right["retrieval_score"]),
    ]


@dataclass(frozen=True)
class FrozenMultiParentEvidenceModel:
    edge: LogisticCalibration
    cardinality: LogisticCalibration
    pair: LogisticCalibration | None
    max_k: int

    def as_dict(self) -> dict:
        return {
            "schema_version": "multiparent_evidence.1",
            "max_k": self.max_k,
            "edge": self.edge.as_dict(),
            "cardinality": self.cardinality.as_dict(),
            "pair": self.pair.as_dict() if self.pair is not None else None,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "FrozenMultiParentEvidenceModel":
        if payload.get("schema_version") != "multiparent_evidence.1":
            raise ValueError("unsupported multi-parent evidence model")
        return cls(
            edge=LogisticCalibration.from_dict(payload["edge"]),
            cardinality=LogisticCalibration.from_dict(payload["cardinality"]),
            pair=(
                LogisticCalibration.from_dict(payload["pair"])
                if payload.get("pair") is not None
                else None
            ),
            max_k=int(payload["max_k"]),
        )

    def transform(self, record, score_map: Mapping[PairRef, float]) -> dict:
        candidates: list[DagCandidate] = []
        cardinality_scores: dict[str, dict[int, float]] = {}
        pair_scores: dict[PairKey, float] = {}
        edge_logits_by_child: dict[int, list[tuple[dict, float]]] = {}
        for child in range(len(record.nodes)):
            rows = _ranked_rows(record, score_map, child, self.max_k)
            raw = [[row["raw_score"]] for row in rows]
            logits = self.edge.log_odds(raw).tolist() if rows else []
            edge_logits_by_child[child] = list(zip(rows, logits))
            for row, logit in edge_logits_by_child[child]:
                candidates.append(
                    DagCandidate(
                        record.nodes[child], row["parent"], float(logit)
                    )
                )
            features = _cardinality_features(logits)
            cardinality_logit = float(self.cardinality.log_odds([features])[0])
            cardinality_scores[record.nodes[child]] = {
                0: 0.0,
                1: 0.0,
                2: cardinality_logit,
            }
            if self.pair is not None:
                pair_rows = sorted(
                    edge_logits_by_child[child], key=lambda item: -item[1]
                )[:3]
                scored_pairs = []
                for (left, left_logit), (right, right_logit) in combinations(
                    pair_rows, 2
                ):
                    features = _pair_features(
                        left, right, left_logit, right_logit
                    )
                    pair_logit = float(self.pair.log_odds([features])[0])
                    scored_pairs.append(
                        (left, left_logit, right, right_logit, pair_logit)
                    )
                if scored_pairs:
                    best_single = max(logits)
                    best_pair_logit = max(item[-1] for item in scored_pairs)
                    for left, left_logit, right, right_logit, pair_logit in scored_pairs:
                        key = (
                            record.nodes[child],
                            *sorted((left["parent"], right["parent"])),
                        )
                        # Pair evidence ranks sets but does not decide cardinality:
                        # the best pair ties the best single before the learned
                        # cardinality log-odds and structural penalty are applied.
                        pair_scores[key] = (
                            best_single
                            + pair_logit
                            - best_pair_logit
                            - left_logit
                            - right_logit
                        )
        multi_parent_candidate_edges = {
            edge
            for child, left, right in pair_scores
            for edge in ((child, left), (child, right))
        }
        return {
            "candidates": candidates,
            "cardinality_scores": cardinality_scores,
            "pair_scores": pair_scores,
            "multi_parent_candidate_edges": multi_parent_candidate_edges,
        }

    def conditional_residual_ranking(
        self,
        record,
        score_map: Mapping[PairRef, float],
        child_index: int,
        primary_parent: str,
        *,
        base_k: int,
        max_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rank candidates beyond base_k conditioned on the chosen first parent."""
        limit = min(max_k or self.max_k, len(record.retrieved[child_index]))
        rows = _ranked_rows(record, score_map, child_index, limit)
        primary = next((row for row in rows if row["parent"] == primary_parent), None)
        if primary is None:
            raise ValueError("primary parent is absent from the candidate pool")
        logits = self.edge.log_odds([[row["raw_score"]] for row in rows]).tolist()
        primary_index = rows.index(primary)
        primary_logit = logits[primary_index]
        ranked = []
        for row, logit in zip(rows[base_k:], logits[base_k:]):
            if row["parent"] == primary_parent:
                continue
            if self.pair is None:
                utility = float(logit)
            else:
                utility = float(
                    self.pair.log_odds(
                        [_pair_features(primary, row, primary_logit, logit)]
                    )[0]
                )
            ranked.append((row["parent"], utility))
        return sorted(ranked, key=lambda item: (-item[1], item[0]))


def fit_frozen_multiparent_evidence(
    records,
    score_maps: list[Mapping[PairRef, float]],
    *,
    max_k: int,
    l2: float = 1.0,
) -> tuple[FrozenMultiParentEvidenceModel, dict]:
    """Fit edge calibration, a cardinality head, and a conditional pair head."""
    if len(records) != len(score_maps):
        raise ValueError("records and score maps must align")
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
            parent for gold_child, parent in record.gold if gold_child == record.nodes[child]
        }
        logits = edge.log_odds([[row["raw_score"]] for row in rows]).tolist() if rows else []
        cardinality_features.append(_cardinality_features(logits))
        cardinality_labels.append(int(len(gold_parents) >= 2))
        for (left, left_logit), (right, right_logit) in combinations(
            zip(rows, logits), 2
        ):
            pair_features.append(
                _pair_features(left, right, left_logit, right_logit)
            )
            pair_labels.append(
                int({left["parent"], right["parent"]}.issubset(gold_parents))
            )
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
        "edge": calibration_metrics(
            edge_labels, edge.probabilities(edge_features)
        ),
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
    }
    return model, diagnostics
