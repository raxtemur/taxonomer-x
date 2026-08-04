"""Fit and apply temperature+bias calibration for learned ROOT logits."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def apply_root_calibration(logits: torch.Tensor, calibration: dict | None) -> torch.Tensor:
    if not calibration:
        return logits
    temperature = max(float(calibration.get("temperature", 1.0)), 1e-4)
    return logits / temperature + float(calibration.get("bias", 0.0))
def root_decision_margins(
    root_scores: torch.Tensor,
    child_index: torch.Tensor,
    edge_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ROOT-vs-best-edge margins and best edge score for every node."""
    if root_scores.ndim != 1 or child_index.ndim != 1 or edge_scores.ndim != 1:
        raise ValueError("root_scores, child_index and edge_scores must be 1-D")
    if child_index.numel() != edge_scores.numel():
        raise ValueError("child_index and edge_scores must be aligned")
    best = root_scores.new_full(root_scores.shape, -torch.inf)
    if child_index.numel():
        if hasattr(best, "scatter_reduce_"):
            best.scatter_reduce_(0, child_index.long(), edge_scores, reduce="amax", include_self=True)
        else:  # pragma: no cover - old torch compatibility
            for index, value in zip(child_index.tolist(), edge_scores):
                best[index] = torch.maximum(best[index], value)
    best = torch.where(torch.isfinite(best), best, root_scores)
    return root_scores - best, best


def apply_root_margin_calibration(
    root_scores: torch.Tensor,
    child_index: torch.Tensor,
    edge_scores: torch.Tensor,
    calibration: dict | None,
) -> torch.Tensor:
    """Map a calibrated decision margin back to an absolute ROOT potential.

    The threshold is folded into zero so both independent and exact decoders
    compare the same dev-frozen decision scale without a special code path.
    """
    if not calibration:
        return root_scores
    margins, best = root_decision_margins(root_scores, child_index, edge_scores)
    temperature = max(float(calibration.get("temperature", 1.0)), 1e-4)
    shifted = margins / temperature + float(calibration.get("bias", 0.0))
    shifted = shifted - float(calibration.get("threshold", 0.0))
    return best + shifted



def _best_f1_threshold(logits: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    candidates = torch.cat((logits.unique(), logits.new_tensor([-math.inf, math.inf])))
    best_threshold, best_f1 = 0.0, -1.0
    for threshold in candidates:
        pred = logits >= threshold
        gold = labels > 0.5
        tp = int((pred & gold).sum())
        fp = int((pred & ~gold).sum())
        fn = int((~pred & gold).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    return best_threshold, best_f1


def fit_root_calibration(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> dict:
    """Fit on a held-out split; never fit this object on test-domain labels."""
    logits = logits.detach().float().flatten()
    labels = labels.detach().float().flatten()
    if logits.numel() != labels.numel() or not logits.numel():
        raise ValueError("logits and labels must be non-empty aligned tensors")
    log_temperature = torch.zeros((), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature, bias], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        calibrated = logits / log_temperature.exp().clamp_min(1e-4) + bias
        loss = F.binary_cross_entropy_with_logits(calibrated, labels)
        loss.backward()
        return loss

    before = float(F.binary_cross_entropy_with_logits(logits, labels))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-4, 1e4))
    fitted_bias = float(bias.detach())
    calibrated = logits / temperature + fitted_bias
    threshold, f1 = _best_f1_threshold(calibrated, labels)
    return {
        "temperature": temperature,
        "bias": fitted_bias,
        "threshold": threshold,
        "dev_f1": f1,
        "bce_before": before,
        "bce_after": float(F.binary_cross_entropy_with_logits(calibrated, labels)),
        "n_examples": logits.numel(),
    }
def fit_root_margin_calibration(margins: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> dict:
    """Fit the actual decoder decision, ROOT minus the best candidate edge."""
    result = fit_root_calibration(margins, labels, max_iter=max_iter)
    result.update({
        "schema_version": "taxobranch.root_margin_calibration.1",
        "decision_variable": "root_potential_minus_best_candidate_edge",
        "threshold_folded_into_decoder_zero": True,
        "fit_scope": "wordnet_dev_only",
    })
    return result




def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="JSONL: root_logit, is_root")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines() if line]
    calibration = fit_root_calibration(
        torch.tensor([row["root_logit"] for row in rows]),
        torch.tensor([row["is_root"] for row in rows]),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
