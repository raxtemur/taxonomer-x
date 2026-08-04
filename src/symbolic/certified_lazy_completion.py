"""Risk-controlled lazy completion for exact maximum-branching inference.

The deterministic mode uses the analytic upper bound ``edge_potential <= 0``
of :class:`RelationCrossEncoder`.  The conformal mode replaces that loose bound
with a split-development upper envelope.  In either case the stopping test is
exact conditional on the supplied upper bounds: if the optimistic completion
cannot beat the best structure over scored edges, no unscored edge can change
the returned branching while all bounds hold.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from src.symbolic.maximum_branching import BranchCandidate, BranchingResult, maximum_branching


PairRef = tuple[int, int]


@dataclass(frozen=True)
class LinearUpperEnvelope:
    coefficients: tuple[float, ...]
    conformal_offset: float
    alpha: float
    fit_graphs: int
    calibration_graphs: int

    def predict(self, features: list[float]) -> float:
        value = float(np.dot(np.asarray(self.coefficients), np.asarray(features)))
        # Every relation objective used here is a log-probability minus
        # non-negative penalties, hence zero remains a deterministic cap.
        return min(0.0, value + self.conformal_offset)

    def as_dict(self) -> dict:
        return {
            "coefficients": list(self.coefficients),
            "conformal_offset": self.conformal_offset,
            "alpha": self.alpha,
            "fit_graphs": self.fit_graphs,
            "calibration_graphs": self.calibration_graphs,
            "analytic_cap": 0.0,
        }


def stable_fit_split(key: str) -> bool:
    """Deterministically assign a development graph to fit vs calibration."""
    return hashlib.sha256(key.encode("utf-8")).digest()[0] % 2 == 0


def pair_features(retrieved: list[list[tuple[int, float]]], child: int, rank: int, max_k: int) -> list[float]:
    row = retrieved[child]
    finite = [float(score) for _, score in row if math.isfinite(float(score))]
    floor = min(finite, default=0.0)
    score = float(row[rank][1])
    if not math.isfinite(score):
        score = floor
    boundary_rank = min(9, len(row) - 1)
    boundary = float(row[boundary_rank][1]) if boundary_rank >= 0 else floor
    if not math.isfinite(boundary):
        boundary = floor
    return [1.0, score, rank / max(1, max_k - 1), boundary, boundary - score]


def fit_upper_envelope(
    records,
    score_maps: list[dict[PairRef, float]],
    *,
    base_k: int,
    max_k: int,
    alpha: float,
    ridge: float = 1e-6,
) -> LinearUpperEnvelope:
    """Fit on one split and calibrate graph-wise maximum residuals on the other."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    fit_indices = [index for index, row in enumerate(records) if stable_fit_split(row.key)]
    calibration_indices = [index for index in range(len(records)) if index not in fit_indices]
    if not fit_indices or not calibration_indices:
        raise ValueError("development split must contain fit and calibration graphs")
    features: list[list[float]] = []
    targets: list[float] = []
    for index in fit_indices:
        record, scores = records[index], score_maps[index]
        for child, row in enumerate(record.retrieved):
            for rank in range(base_k, min(max_k, len(row))):
                features.append(pair_features(record.retrieved, child, rank, max_k))
                targets.append(float(scores[(child, rank)]))
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
    maxima = []
    for index in calibration_indices:
        record, scores = records[index], score_maps[index]
        residuals = []
        for child, row in enumerate(record.retrieved):
            for rank in range(base_k, min(max_k, len(row))):
                prediction = float(np.dot(coefficients, pair_features(record.retrieved, child, rank, max_k)))
                residuals.append(float(scores[(child, rank)]) - prediction)
        maxima.append(max(residuals, default=0.0))
    ordered = np.sort(np.asarray(maxima, dtype=np.float64))
    quantile_rank = min(len(ordered), math.ceil((len(ordered) + 1) * (1.0 - alpha)))
    offset = float(ordered[max(0, quantile_rank - 1)])
    return LinearUpperEnvelope(
        coefficients=tuple(float(value) for value in coefficients),
        conformal_offset=offset,
        alpha=alpha,
        fit_graphs=len(fit_indices),
        calibration_graphs=len(calibration_indices),
    )


def _candidates(record, refs: set[PairRef], values: dict[PairRef, float]) -> list[BranchCandidate]:
    return [
        BranchCandidate(
            record.nodes[child],
            record.nodes[record.retrieved[child][rank][0]],
            float(values[(child, rank)]),
        )
        for child, rank in sorted(refs)
    ]


def _solve(record, refs: set[PairRef], values: dict[PairRef, float], root_scores: dict[str, float]) -> BranchingResult:
    return maximum_branching(record.nodes, _candidates(record, refs, values), root_scores)


def simulate_lazy_completion(
    record,
    score_map: dict[PairRef, float],
    root_scores: dict[str, float],
    *,
    base_k: int,
    max_k: int,
    envelope: LinearUpperEnvelope | None,
) -> dict:
    """Simulate threat-directed scoring and issue an upper-bound certificate."""
    full_refs = {
        (child, rank)
        for child, row in enumerate(record.retrieved)
        for rank in range(min(max_k, len(row)))
    }
    known = {
        (child, rank)
        for child, row in enumerate(record.retrieved)
        for rank in range(min(base_k, len(row)))
    }
    upper = {}
    for child, rank in full_refs - known:
        upper[(child, rank)] = (
            0.0
            if envelope is None
            else envelope.predict(pair_features(record.retrieved, child, rank, max_k))
        )
    violations = sum(float(score_map[ref]) > upper[ref] + 1e-9 for ref in full_refs - known)
    rounds = 0
    certified = False
    while True:
        current = _solve(record, known, score_map, root_scores)
        optimistic_values = {ref: (score_map[ref] if ref in known else upper[ref]) for ref in full_refs}
        optimistic = _solve(record, full_refs, optimistic_values, root_scores)
        if optimistic.score <= current.score + 1e-9:
            certified = True
            break
        threats = set()
        parent_to_ref = {
            (record.nodes[child], record.nodes[record.retrieved[child][rank][0]]): (child, rank)
            for child, rank in full_refs - known
        }
        for child, parent in optimistic.parents.items():
            ref = parent_to_ref.get((child, parent))
            if ref is not None:
                threats.add(ref)
        if not threats:
            # Numeric ties can leave no explicit unseen edge in the optimistic
            # solution. Scoring the best remaining bound guarantees progress.
            remaining = full_refs - known
            if not remaining:
                break
            threats.add(max(remaining, key=lambda ref: (upper[ref], -ref[0], -ref[1])))
        known.update(threats)
        rounds += 1
        if known == full_refs:
            certified = True
            break
    result = _solve(record, known, score_map, root_scores)
    full = _solve(record, full_refs, score_map, root_scores)
    return {
        "parents": result.parents,
        "full_parents": full.parents,
        "matches_full": result.parents == full.parents,
        "certified": certified,
        "pairs_scored": len(known),
        "full_pairs": len(full_refs),
        "pair_fraction": len(known) / max(1, len(full_refs)),
        "rounds": rounds,
        "upper_bound_violations": int(violations),
        "all_upper_bounds_hold": violations == 0,
        "objective_score": result.score,
        "full_objective_score": full.score,
    }
