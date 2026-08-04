"""Compact adapters for batched symbolic decoding of frozen WordNet caches."""

from __future__ import annotations

from collections import defaultdict


def scored_payload(graph: dict) -> dict:
    child = graph["child_index"].long().tolist()
    parent = graph["parent_index"].long().tolist()
    local = graph["local_scores"].float().tolist()
    retrieval_tensor = graph.get("retrieval_scores", graph["local_scores"])
    retrieval = retrieval_tensor.float().tolist()
    by_child = defaultdict(list)
    for edge, (child_index, parent_index) in enumerate(zip(child, parent)):
        by_child[child_index].append({
            "parent": graph["nodes"][parent_index],
            "retrieval_score": retrieval[edge],
            "local_score": local[edge],
            "context_delta": 0.0,
        })
    roots = graph["root_logits"].float().tolist()
    return {
        "dataset": graph["subgraph_id"],
        "proposals": [
            {"child": node, "root_logit": roots[index], "candidates": by_child[index]}
            for index, node in enumerate(graph["nodes"])
        ],
    }
