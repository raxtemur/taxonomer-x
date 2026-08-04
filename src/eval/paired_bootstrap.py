"""Vectorized paired child-level bootstrap confidence interval for Edge-F1 deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _f1(pred: set[tuple[str, str]], gold: set[tuple[str, str]]) -> float:
    return 2 * len(pred & gold) / max(1, len(pred) + len(gold))


def _paired_bootstrap_units(
    children: list[str],
    a_by_child: dict[str, set[str]],
    b_by_child: dict[str, set[str]],
    gold_by_child: dict[str, set[str]],
    *,
    metric: str,
    samples: int,
    seed: int,
    batch_size: int,
) -> dict:
    if samples < 1 or batch_size < 1:
        raise ValueError("samples and batch_size must be positive")
    if not children:
        return {
            "metric": metric, "delta_b_minus_a": 0.0, "ci95": [0.0, 0.0],
            "probability_positive": 0.0, "samples": samples, "seed": seed,
            "n_children": 0, "implementation": "batched_sufficient_statistics",
            "batch_size": batch_size,
        }
    gold_count = np.asarray([len(gold_by_child.get(child, set())) for child in children], dtype=np.float64)
    a_count = np.asarray([len(a_by_child.get(child, set())) for child in children], dtype=np.float64)
    b_count = np.asarray([len(b_by_child.get(child, set())) for child in children], dtype=np.float64)
    a_tp = np.asarray([len(a_by_child.get(child, set()) & gold_by_child.get(child, set())) for child in children], dtype=np.float64)
    b_tp = np.asarray([len(b_by_child.get(child, set()) & gold_by_child.get(child, set())) for child in children], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        chosen = rng.integers(0, len(children), size=(size, len(children)))
        sampled_gold = gold_count[chosen].sum(axis=1)
        sampled_a = a_count[chosen].sum(axis=1)
        sampled_b = b_count[chosen].sum(axis=1)
        f1_a = 2.0 * a_tp[chosen].sum(axis=1) / np.maximum(1.0, sampled_a + sampled_gold)
        f1_b = 2.0 * b_tp[chosen].sum(axis=1) / np.maximum(1.0, sampled_b + sampled_gold)
        deltas[start : start + size] = f1_b - f1_a
    observed_a = 2.0 * a_tp.sum() / max(1.0, a_count.sum() + gold_count.sum())
    observed_b = 2.0 * b_tp.sum() / max(1.0, b_count.sum() + gold_count.sum())
    return {
        "metric": metric,
        "delta_b_minus_a": float(observed_b - observed_a),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "probability_positive": float(np.mean(deltas > 0)),
        "samples": samples,
        "seed": seed,
        "n_children": len(children),
        "implementation": "batched_sufficient_statistics",
        "batch_size": batch_size,
    }


def _direct_units(edges: set[tuple[str, str]], children: list[str]) -> dict[str, set[str]]:
    result = {child: set() for child in children}
    for child, parent in edges:
        result.setdefault(child, set()).add(parent)
    return result


def _ancestor_units(edges: set[tuple[str, str]], children: list[str]) -> dict[str, set[str]]:
    adjacency = {child: [] for child in children}
    for child, parent in edges:
        adjacency.setdefault(child, []).append(parent)
        adjacency.setdefault(parent, [])
    result = {}
    for child in children:
        seen, stack = set(), list(adjacency.get(child, ()))
        while stack:
            parent = stack.pop()
            if parent == child or parent in seen:
                continue
            seen.add(parent)
            stack.extend(adjacency.get(parent, ()))
        result[child] = seen
    return result


def paired_bootstrap(
    pred_a: set[tuple[str, str]],
    pred_b: set[tuple[str, str]],
    gold: set[tuple[str, str]],
    samples: int = 10000,
    seed: int = 42,
    batch_size: int = 256,
) -> dict:
    """Paired child bootstrap for direct Edge-F1."""
    children = sorted({child for child, _ in gold | pred_a | pred_b})
    return _paired_bootstrap_units(
        children, _direct_units(pred_a, children), _direct_units(pred_b, children),
        _direct_units(gold, children), metric="edge_f1", samples=samples,
        seed=seed, batch_size=batch_size,
    )


def paired_ancestor_bootstrap(
    pred_a: set[tuple[str, str]],
    pred_b: set[tuple[str, str]],
    gold: set[tuple[str, str]],
    universe: set[str],
    samples: int = 10000,
    seed: int = 42,
    batch_size: int = 256,
) -> dict:
    """Paired child bootstrap for transitive Ancestor-F1."""
    children = sorted(set(universe) | {node for edge in gold | pred_a | pred_b for node in edge})
    return _paired_bootstrap_units(
        children, _ancestor_units(pred_a, children), _ancestor_units(pred_b, children),
        _ancestor_units(gold, children), metric="ancestor_f1", samples=samples,
        seed=seed, batch_size=batch_size,
    )


def _edges(path: Path) -> set[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {(row["child"], row["parent"]) for row in payload["edges"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = paired_bootstrap(
        _edges(args.a), _edges(args.b), _edges(args.gold), args.samples, args.seed,
        args.batch_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
