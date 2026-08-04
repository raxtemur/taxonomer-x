"""Controlled decoder ablations over one shared scored candidate set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import torch

from src.eval.root_calibration import apply_root_calibration
from src.symbolic.maximum_branching import ROOT, BranchCandidate, maximum_branching


def _root_score(row: dict, calibration: dict | None) -> float:
    return float(apply_root_calibration(torch.tensor(float(row.get("root_logit", 0.0))), calibration))


def _edge_score(item: dict, stage: str) -> float:
    if stage == "embedding": return float(item.get("retrieval_score", 0.0))
    if stage == "relation": return float(item.get("local_score", 0.0))
    if stage == "context":
        if "local_score" in item: return float(item["local_score"]) + float(item.get("context_delta", 0.0))
        return float(item.get("root_relative_score", 0.0))
    raise ValueError(stage)


def _variant(name: str, parents: dict[str, str | None], scores: dict[str, float]) -> dict:
    edges = [(child, parent) for child, parent in parents.items() if parent is not None]
    graph = nx.DiGraph(); graph.add_nodes_from(parents); graph.add_edges_from(edges)
    return {"system": name,
            "edges": [{"child": c, "parent": p, "score": scores[c]} for c, p in sorted(edges)],
            "roots": sorted(c for c, p in parents.items() if p is None),
            "graph_stats": {"is_dag": nx.is_directed_acyclic_graph(graph),
                            "n_roots": sum(p is None for p in parents.values()),
                            "n_nodes": len(parents), "n_edges": len(edges)}}


def _top1(payload: dict, stage: str, compare_root: bool, calibration: dict | None) -> dict:
    parents = {}; scores = {}
    for row in payload.get("proposals", []):
        child = row["child"]; candidates = row.get("candidates", [])
        best = max(candidates, key=lambda item: _edge_score(item, stage)) if candidates else None
        if best is None: parents[child], scores[child] = None, _root_score(row, calibration); continue
        edge_score = _edge_score(best, stage); root_score = _root_score(row, calibration)
        parents[child] = None if compare_root and root_score >= edge_score else best["parent"]
        scores[child] = root_score if parents[child] is None else edge_score
    return {"parents": parents, "scores": scores}


def _repair_cycles(parents: dict[str, str | None], scores: dict[str, float]) -> None:
    while True:
        graph = nx.DiGraph((child, parent) for child, parent in parents.items() if parent is not None)
        try: cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle: return
        parents[min((edge[0] for edge in cycle), key=lambda node: scores[node])] = None


def decode_variants(payload: dict, calibration: dict | None = None) -> dict:
    variants = {}
    specifications = (("embedding_top1", "embedding", False), ("relation_top1", "relation", False),
                      ("relation_root_top1", "relation", True), ("context_top1", "context", True))
    for name, stage, compare_root in specifications:
        state = _top1(payload, stage, compare_root, calibration)
        variants[name] = _variant(name, state["parents"], state["scores"])
    greedy = _top1(payload, "context", True, calibration); _repair_cycles(greedy["parents"], greedy["scores"])
    variants["greedy_cycle_repair"] = _variant("greedy_cycle_repair", greedy["parents"], greedy["scores"])
    rows = payload.get("proposals", []); nodes = {row["child"] for row in rows}
    candidates = [BranchCandidate(row["child"], item["parent"], _edge_score(item, "context"))
                  for row in rows for item in row.get("candidates", []) if item["parent"] in nodes]
    root_scores = {row["child"]: _root_score(row, calibration) for row in rows}
    result = maximum_branching(nodes, candidates, root_scores)
    parents = {child: (None if parent == ROOT else parent) for child, parent in result.parents.items()}
    scores = {child: (root_scores[child] if parent is None else
                      next(c.score for c in candidates if c.child == child and c.parent == parent))
              for child, parent in parents.items()}
    variants["exact_branching"] = _variant("exact_branching", parents, scores)
    return {"dataset": payload.get("dataset") or payload.get("domain"), "root_calibration": calibration,
            "shared_candidate_set": True, "variants": variants}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True); parser.add_argument("--root-calibration", type=Path)
    parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    payload = json.loads(args.scored.read_text(encoding="utf-8"))
    calibration = json.loads(args.root_calibration.read_text()) if args.root_calibration else None
    result = decode_variants(payload, calibration); args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
