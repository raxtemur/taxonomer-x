"""Tune the symbolic policy exclusively on node-disjoint WordNet development graphs."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch

from src.eval.texeval_metrics import ancestor_metrics, edge_metrics
from src.experiments.symbolic_cached import scored_payload
from src.symbolic.confidence_repair import (
    ROOT,
    SymbolicPolicy,
    _choices,
    exact_parents,
    independent_parents,
)


def _decode(payload: dict, policy: SymbolicPolicy) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    choices, roots = _choices(
        payload, stage="relation", weights=None, learned_root=True,
        root_penalty=policy.root_penalty,
    )
    reference, _, margins = independent_parents(choices, roots, compare_root=True)
    decoded, _, _ = exact_parents(
        choices, roots, policy=policy, reference=reference, margins=margins,
        use_edit_penalty=True, use_root_budget=True,
    )
    baseline_edges = {(child, parent) for child, parent in reference.items() if parent != ROOT}
    decoded_edges = {(child, parent) for child, parent in decoded.items() if parent != ROOT}
    return baseline_edges, decoded_edges


def evaluate_policy(graphs: list[dict], gold: dict[str, dict], policy: SymbolicPolicy) -> dict:
    edge_values = []
    ancestor_values = []
    baseline_edge_values = []
    baseline_ancestor_values = []
    valid = []
    root_counts = []
    for graph in graphs:
        payload = scored_payload(graph)
        baseline, decoded = _decode(payload, policy)
        row = gold[graph["subgraph_id"]]
        gold_edges = {tuple(edge) for edge in row["edges"]}
        universe = set(row["nodes"])
        baseline_edge_values.append(edge_metrics(baseline, gold_edges)["f1"])
        baseline_ancestor_values.append(ancestor_metrics(baseline, gold_edges, universe)["f1"])
        edge_values.append(edge_metrics(decoded, gold_edges)["f1"])
        ancestor_values.append(ancestor_metrics(decoded, gold_edges, universe)["f1"])
        graph_roots = universe - {child for child, _ in decoded}
        root_counts.append(len(graph_roots))
        valid.append(True)
    mean = lambda values: sum(values) / max(1, len(values))
    return {
        "macro_edge_f1": mean(edge_values),
        "macro_ancestor_f1": mean(ancestor_values),
        "baseline_macro_edge_f1": mean(baseline_edge_values),
        "baseline_macro_ancestor_f1": mean(baseline_ancestor_values),
        "valid_rate": mean(valid),
        "mean_roots": mean(root_counts),
        "n_subgraphs": len(graphs),
    }


def tune(
    graphs: list[dict],
    gold: dict[str, dict],
    *,
    edge_floor_drop: float,
    edit_bases: tuple[float, ...],
    edit_scales: tuple[float, ...],
    root_fractions: tuple[float, ...],
) -> dict:
    rows = []
    for edit_base, edit_scale, root_fraction in itertools.product(edit_bases, edit_scales, root_fractions):
        policy = SymbolicPolicy(
            edit_base=edit_base,
            edit_scale=edit_scale,
            max_root_fraction=root_fraction,
        )
        metrics = evaluate_policy(graphs, gold, policy)
        edge_floor = metrics["baseline_macro_edge_f1"] - edge_floor_drop
        eligible = metrics["macro_edge_f1"] >= edge_floor and metrics["valid_rate"] == 1.0
        objective = metrics["macro_ancestor_f1"] + 0.25 * metrics["macro_edge_f1"]
        rows.append({
            "policy": asdict(policy),
            "metrics": metrics,
            "eligible": eligible,
            "selection_objective": objective if eligible else -math.inf,
        })
    rows.sort(
        key=lambda row: (
            -row["selection_objective"],
            row["policy"]["edit_scale"],
            row["policy"]["edit_base"],
            row["policy"]["max_root_fraction"],
        )
    )
    if not rows or not rows[0]["eligible"]:
        raise RuntimeError("No symbolic policy satisfies the dev edge-F1 floor")
    best = rows[0]
    return {
        "schema_version": "taxobranch.symbolic_policy.1",
        "fit_split": "wordnet_node_disjoint_dev",
        "test_labels_used": False,
        "selection": "max ancestor-F1 + 0.25 edge-F1 subject to validity=1 and edge-F1 floor",
        "edge_floor_drop": edge_floor_drop,
        "policy": best["policy"],
        "dev_metrics": best["metrics"],
        "grid_size": len(rows),
        "leaderboard": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--subgraphs", type=Path,
        default=Path("data/processed/taxobranch/wordnet_relations/subgraphs/dev.jsonl"),
    )
    parser.add_argument("--edge-floor-drop", type=float, default=0.01)
    parser.add_argument("--edit-bases", type=float, nargs="+", default=(0.0, 0.10, 0.25))
    parser.add_argument("--edit-scales", type=float, nargs="+", default=(0.0, 0.50, 1.0, 2.0))
    parser.add_argument("--root-fractions", type=float, nargs="+", default=(0.05, 0.10, 0.20, 1.0))
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    graphs = cache["graphs"][: args.max_graphs or None]
    gold = {
        row["subgraph_id"]: row
        for row in (
            json.loads(line)
            for line in args.subgraphs.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    result = tune(
        graphs, gold, edge_floor_drop=args.edge_floor_drop,
        edit_bases=tuple(args.edit_bases), edit_scales=tuple(args.edit_scales),
        root_fractions=tuple(args.root_fractions),
    )
    result["score_cache"] = str(args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "policy": result["policy"], "dev": result["dev_metrics"]}))


if __name__ == "__main__":
    main()
