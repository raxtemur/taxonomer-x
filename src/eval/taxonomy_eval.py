"""Unified relation/structure evaluation for TExEval and generic subgraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx

from src.data.texeval_loader import gold_edges_as_set, load_terms
from src.eval.texeval_metrics import ancestor_metrics, edge_metrics


def structural_metrics(edges: set[tuple[str, str]], universe: set[str]) -> dict:
    graph = nx.DiGraph()
    graph.add_nodes_from(universe)
    graph.add_edges_from(edges)
    parent_counts = {}
    for child, _ in edges:
        parent_counts[child] = parent_counts.get(child, 0) + 1
    roots = universe - {child for child, _ in edges}
    undirected = graph.to_undirected()
    is_dag = nx.is_directed_acyclic_graph(graph)
    single_parent = all(count <= 1 for count in parent_counts.values())
    return {
        "is_dag": is_dag,
        "dag_valid": is_dag,
        "single_parent": single_parent,
        "max_parent_count": max(parent_counts.values(), default=0),
        "all_nodes_covered": set(graph) == universe,
        "n_roots": len(roots),
        "n_components": nx.number_connected_components(undirected) if universe else 0,
        "valid": is_dag and single_parent,
    }


def evaluate_edges(
    pred: set[tuple[str, str]], gold: set[tuple[str, str]], universe: set[str]
) -> dict:
    return {
        "edge": edge_metrics(pred, gold),
        "ancestor": ancestor_metrics(pred, gold, universe),
        "structure": structural_metrics(pred, universe),
        "n_pred_edges": len(pred),
        "n_gold_edges": len(gold),
    }


def evaluate_decoded(
    payload: dict, gold: set[tuple[str, str]], universe: set[str]
) -> dict:
    results = {}
    variants = payload.get("variants") or {payload.get("system", "system"): payload}
    for name, variant in variants.items():
        pred = {(row["child"], row["parent"]) for row in variant.get("edges", [])}
        results[name] = evaluate_edges(pred, gold, universe)
    return {
        "dataset": payload.get("dataset") or payload.get("domain"),
        "systems": results,
    }


def load_subgraph(
    path: Path, subgraph_id: str | None
) -> tuple[set[tuple[str, str]], set[str]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if subgraph_id is not None:
        rows = [row for row in rows if row.get("subgraph_id") == subgraph_id]
    if len(rows) != 1:
        raise ValueError(
            "Specify --subgraph-id when the gold JSONL contains multiple subgraphs"
        )
    row = rows[0]
    return {tuple(edge) for edge in row["edges"]}, set(row["nodes"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--texeval-domain")
    group.add_argument("--gold-subgraphs", type=Path)
    parser.add_argument("--subgraph-id")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    if args.texeval_domain:
        gold = gold_edges_as_set(args.texeval_domain)
        universe = set(load_terms(args.texeval_domain))
    else:
        gold, universe = load_subgraph(args.gold_subgraphs, args.subgraph_id)
    result = evaluate_decoded(payload, gold, universe)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
