"""Taxonomy-specific error categories for direct-parent predictions."""

from __future__ import annotations
import argparse, json
from collections import Counter, defaultdict, deque
from pathlib import Path


def _ancestor_distances(edges: set[tuple[str, str]]) -> dict[str, dict[str, int]]:
    parents = defaultdict(set)
    for child, parent in edges: parents[child].add(parent)
    output = {}
    for child in {node for edge in edges for node in edge}:
        distance = {}; queue = deque((parent, 1) for parent in parents[child])
        while queue:
            node, depth = queue.popleft()
            if node in distance and distance[node] <= depth: continue
            distance[node] = depth; queue.extend((parent, depth+1) for parent in parents[node])
        output[child] = distance
    return output


def classify_errors(pred: set[tuple[str, str]], gold: set[tuple[str, str]], universe: set[str] | None = None) -> dict:
    universe = universe or {node for edge in gold | pred for node in edge}; pred_parent = defaultdict(set); gold_parent = defaultdict(set)
    for child, parent in pred: pred_parent[child].add(parent)
    for child, parent in gold: gold_parent[child].add(parent)
    distances = _ancestor_distances(gold); counts = Counter(); examples = defaultdict(list)
    for child in sorted(universe):
        guesses, truths = pred_parent[child], gold_parent[child]
        if guesses & truths: category = "correct"
        elif not guesses and truths: category = "root_error"
        elif guesses and not truths: category = "spurious_parent_on_gold_root"
        else:
            parent = sorted(guesses)[0] if guesses else None; distance = distances.get(child, {}).get(parent)
            if distance == 2: category = "ancestor_instead_of_direct"
            elif distance is not None and distance > 2: category = "generic_parent_bias"
            elif parent is not None and child in distances.get(parent, {}): category = "reverse_relation"
            elif parent is not None and any(gold_parent[parent] & gold_parent[truth] for truth in truths): category = "sibling_confusion"
            else: category = "unrelated_or_other"
        counts[category] += 1
        if len(examples[category]) < 20: examples[category].append({"child": child, "predicted": sorted(guesses), "gold": sorted(truths)})
    return {"counts": dict(counts), "rates": {key: value/max(1, len(universe)) for key, value in counts.items()}, "examples": dict(examples), "n_nodes": len(universe)}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--pred", type=Path, required=True); p.add_argument("--gold", type=Path, required=True); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    read = lambda path: {(r["child"], r["parent"]) for r in json.loads(path.read_text())["edges"]}; result = classify_errors(read(args.pred), read(args.gold))
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__": main()
