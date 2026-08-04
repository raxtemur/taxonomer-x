"""Calibration utilities for variable-cardinality structured decoding.

Single-parent ranking potentials are order-preserving but are not necessarily
additive evidence for selecting an arbitrary number of edges. This module fits
a deterministic regularized logistic calibration and exposes calibrated log
odds, which are the correct independent-Bernoulli additive utilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


def _matrix(features: Iterable[float] | Iterable[Iterable[float]]) -> np.ndarray:
    array = np.asarray(list(features), dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("features must be a non-empty vector or matrix")
    if not np.isfinite(array).all():
        raise ValueError("features must be finite")
    return array


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    output = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    output[~positive] = exp_logits / (1.0 + exp_logits)
    return output


@dataclass(frozen=True)
class LogisticCalibration:
    coefficients: tuple[float, ...]
    intercept: float
    l2: float
    iterations: int
    converged: bool
    n_examples: int
    n_positive: int

    def log_odds(
        self, features: Iterable[float] | Iterable[Iterable[float]]
    ) -> np.ndarray:
        matrix = _matrix(features)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if matrix.shape[1] != coefficients.size:
            raise ValueError(
                f"expected {coefficients.size} features, got {matrix.shape[1]}"
            )
        return matrix @ coefficients + float(self.intercept)

    def probabilities(
        self, features: Iterable[float] | Iterable[Iterable[float]]
    ) -> np.ndarray:
        return _sigmoid(self.log_odds(features))

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LogisticCalibration":
        return cls(
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            l2=float(payload["l2"]),
            iterations=int(payload["iterations"]),
            converged=bool(payload["converged"]),
            n_examples=int(payload["n_examples"]),
            n_positive=int(payload["n_positive"]),
        )


def fit_logistic_calibration(
    features: Iterable[float] | Iterable[Iterable[float]],
    labels: Iterable[int | bool | float],
    *,
    l2: float = 1.0,
    max_iter: int = 200,
    tolerance: float = 1e-9,
) -> LogisticCalibration:
    """Fit an unweighted regularized logistic model with deterministic IRLS."""
    matrix = _matrix(features)
    target = np.asarray(list(labels), dtype=np.float64)
    if target.shape != (matrix.shape[0],):
        raise ValueError("labels must have one value per feature row")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    positives = int(target.sum())
    if positives == 0 or positives == len(target):
        raise ValueError("calibration requires both positive and negative examples")
    if l2 < 0 or max_iter < 1 or tolerance <= 0:
        raise ValueError("invalid optimization parameters")

    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack((normalized, np.ones(len(normalized))))
    beta = np.zeros(design.shape[1], dtype=np.float64)
    beta[-1] = np.log((positives + 0.5) / (len(target) - positives + 0.5))

    def objective(parameters: np.ndarray) -> float:
        logits = design @ parameters
        nll = np.logaddexp(0.0, logits).sum() - target @ logits
        return float(nll + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1]))

    converged = False
    iterations = 0
    for iterations in range(1, max_iter + 1):
        logits = design @ beta
        probability = _sigmoid(logits)
        weight = np.maximum(probability * (1.0 - probability), 1e-9)
        gradient = design.T @ (probability - target)
        gradient[:-1] += l2 * beta[:-1]
        hessian = design.T @ (weight[:, None] * design)
        hessian[:-1, :-1] += l2 * np.eye(design.shape[1] - 1)
        step = np.linalg.solve(hessian, gradient)
        if float(np.max(np.abs(step))) < tolerance:
            converged = True
            break
        old_objective = objective(beta)
        multiplier = 1.0
        while multiplier > 1e-8:
            proposal = beta - multiplier * step
            if objective(proposal) <= old_objective:
                beta = proposal
                break
            multiplier *= 0.5
        else:
            break

    coefficients = beta[:-1] / scale
    intercept = beta[-1] - float(coefficients @ mean)
    return LogisticCalibration(
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(intercept),
        l2=float(l2),
        iterations=iterations,
        converged=converged,
        n_examples=len(target),
        n_positive=positives,
    )


def calibration_metrics(
    labels: Iterable[int | bool | float],
    probabilities: Iterable[float],
    *,
    bins: int = 10,
) -> dict[str, float]:
    target = np.asarray(list(labels), dtype=np.float64)
    probability = np.asarray(list(probabilities), dtype=np.float64)
    if target.shape != probability.shape or target.ndim != 1:
        raise ValueError("labels and probabilities must be aligned vectors")
    if not len(target) or not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("labels must be a non-empty binary vector")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    if bins < 1:
        raise ValueError("bins must be positive")

    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        mask = (probability >= left) & (
            probability <= right if index + 1 == bins else probability < right
        )
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(probability[mask].mean() - target[mask].mean())
            )

    order = np.argsort(-probability, kind="stable")
    ranked = target[order]
    true_positive = np.cumsum(ranked)
    precision = true_positive / np.arange(1, len(ranked) + 1)
    positives = int(target.sum())
    average_precision = (
        float(precision[ranked == 1].sum() / positives) if positives else 0.0
    )
    return {
        "prevalence": float(target.mean()),
        "mean_probability": float(probability.mean()),
        "brier": float(np.mean((probability - target) ** 2)),
        "ece": float(ece),
        "average_precision": average_precision,
    }
