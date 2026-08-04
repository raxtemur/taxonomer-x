"""Evaluate relation/structure variants on node-disjoint cached WordNet subgraphs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.eval.taxonomy_eval import evaluate_decoded
from src.models.candidate_graph import FEATURE_NAMES, ablate_features
from src.models.relation_aware import CandidateGraphContextualizer
from src.symbolic.decoding_baselines import decode_variants


def load_context(path: Path, edge_dim: int, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False); cfg = payload.get("config", {})
    model = CandidateGraphContextualizer(edge_dim, len(FEATURE_NAMES), int(cfg.get("hidden_dim", 256)), float(cfg.get("dropout", .1)))
    model.load_state_dict(payload["state_dict"]); return model.to(device).eval()


def score_cached_graph(graph: dict, model, device, disabled_features=()) -> dict:
    child = graph["child_index"].to(device); parent = graph["parent_index"].to(device)
    features = ablate_features(graph["scalar_features"].to(device), disabled_features)
    with torch.inference_mode():
        delta = model(graph["edge_repr"].float().to(device), child, parent, features, n_nodes=len(graph["nodes"])).cpu()
    local = graph["local_scores"].cpu(); roots = graph["root_logits"].cpu(); by_child = defaultdict(list)
    retrieval = graph.get("retrieval_scores", local).cpu()
    for edge, (c, p) in enumerate(zip(child.cpu().tolist(), parent.cpu().tolist())):
        by_child[c].append({"parent": graph["nodes"][p], "retrieval_score": float(retrieval[edge]),
                            "local_score": float(local[edge]), "context_delta": float(delta[edge]),
                            "root_relative_score": float(local[edge]+delta[edge]-roots[c])})
    return {"dataset": graph["subgraph_id"], "proposals": [
        {"child": node, "root_logit": float(roots[c]), "candidates": by_child[c]}
        for c, node in enumerate(graph["nodes"])
    ]}


def aggregate(rows: list[dict]) -> dict:
    systems = defaultdict(list)
    for row in rows:
        for name, metrics in row["metrics"]["systems"].items(): systems[name].append(metrics)
    return {name: {"n_subgraphs": len(items),
                   "macro_edge_f1": sum(item["edge"]["f1"] for item in items)/len(items),
                   "macro_ancestor_f1": sum(item["ancestor"]["f1"] for item in items)/len(items),
                   "valid_rate": sum(item["structure"]["valid"] for item in items)/len(items)}
            for name, items in systems.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--subgraphs", type=Path, default=Path("data/processed/taxobranch/wordnet_relations/subgraphs/test.jsonl"))
    p.add_argument("--context-checkpoint", type=Path, required=True); p.add_argument("--disable-features", nargs="*", choices=FEATURE_NAMES, default=[])
    p.add_argument("--device", default="auto"); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    graphs = torch.load(args.cache, map_location="cpu", weights_only=False)["graphs"]
    gold_rows = {row["subgraph_id"]: row for row in (json.loads(line) for line in args.subgraphs.read_text().splitlines() if line)}
    model = load_context(args.context_checkpoint, graphs[0]["edge_repr"].shape[-1], device); rows = []
    for graph in graphs:
        gold = gold_rows[graph["subgraph_id"]]; scored = score_cached_graph(graph, model, device, args.disable_features)
        decoded = decode_variants(scored); metrics = evaluate_decoded(decoded, {tuple(edge) for edge in gold["edges"]}, set(gold["nodes"]))
        rows.append({"subgraph_id": graph["subgraph_id"], "metrics": metrics})
    result = {"benchmark": "wordnet_node_disjoint_test", "n_subgraphs": len(rows), "per_subgraph": rows, "aggregate": aggregate(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["aggregate"], sort_keys=True))


if __name__ == "__main__": main()
