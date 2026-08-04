#!/usr/bin/env python3
"""Build five-way relation supervision and graph/root examples from WordNet.

Splitting is performed by WordNet lexicographer file *before* examples are
derived.  No synset or derived path can therefore cross train/dev/test.
External benchmark vocabulary is removed at synset level using every lemma,
not merely a synset's first surface form.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.data.wordnet_graph import SynsetNode, WordNetGraph, load_wordnet_graph


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DICT = REPO / "data" / "raw" / "wordnet" / "dict"
DEFAULT_BENCH = REPO / "data" / "processed" / "taxobranch" / "benchmarks"
DEFAULT_OUT = REPO / "data" / "processed" / "taxobranch" / "wordnet_relations"
RELATIONS = ("direct", "ancestor", "sibling", "reverse", "unrelated")
SPACE_RE = re.compile(r"[\s_\-]+")


def normalize(term: str) -> str:
    return SPACE_RE.sub(" ", term.strip().casefold()).strip()


def stable_id(*parts: str) -> str:
    payload = "\t".join(parts).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=12).hexdigest()


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_blacklist(bench_root: Path) -> tuple[set[str], dict[str, int]]:
    values: set[str] = set()
    counts: dict[str, int] = {}
    for filename in ("texeval2_vocab.txt", "schemaorg30_vocab.txt"):
        path = bench_root / filename
        current = {
            normalize(line) for line in path.read_text(encoding="utf-8").splitlines()
            if normalize(line)
        }
        counts[filename] = len(current)
        values.update(current)
    return values, counts


def clean_graph(graph: WordNetGraph, blacklist: set[str]) -> tuple[WordNetGraph, set[str]]:
    removed = {
        synset_id for synset_id, node in graph.nodes.items()
        if any(normalize(lemma) in blacklist for lemma in node.lemmas)
    }
    return graph.induced(set(graph.nodes) - removed), removed


def assign_domains(graph: WordNetGraph, seed: int) -> tuple[dict[str, str], dict[str, list[str]]]:
    domains: dict[str, list[str]] = defaultdict(list)
    for synset_id, node in graph.nodes.items():
        domains[node.lexname].append(synset_id)
    edge_weight = Counter()
    for child, parent in graph.edges:
        if graph.nodes[child].lexname == graph.nodes[parent].lexname:
            edge_weight[graph.nodes[child].lexname] += 1
    rng = random.Random(seed)
    names = list(domains)
    rng.shuffle(names)
    names.sort(key=lambda name: edge_weight[name], reverse=True)
    target = {"train": 0.80, "dev": 0.10, "test": 0.10}
    totals = {split: 0 for split in target}
    total_weight = max(1, sum(max(1, edge_weight[name]) for name in names))
    domain_split: dict[str, str] = {}
    for name in names:
        weight = max(1, edge_weight[name])
        split = min(target, key=lambda key: totals[key] / (target[key] * total_weight))
        domain_split[name] = split
        totals[split] += weight
    by_split = {
        split: sorted(
            synset_id for synset_id, node in graph.nodes.items()
            if domain_split[node.lexname] == split
        )
        for split in target
    }
    return domain_split, by_split


def pair_row(
    graph: WordNetGraph, child: str, parent: str, relation: str,
    split: str, distance: int | None = None,
) -> dict:
    c, p = graph.nodes[child], graph.nodes[parent]
    return {
        "example_id": stable_id(split, relation, child, parent),
        "split": split,
        "relation": relation,
        "distance": distance,
        "child_synset": child,
        "parent_synset": parent,
        "child": c.surface,
        "parent": p.surface,
        "child_lemmas": list(c.lemmas),
        "parent_lemmas": list(p.lemmas),
        "child_gloss": c.gloss,
        "parent_gloss": p.gloss,
        "child_pos": c.pos,
        "parent_pos": p.pos,
        "lexname": c.lexname,
    }


def relation_rows(graph: WordNetGraph, split: str, seed: int):
    rng = random.Random(seed)
    direct = sorted(graph.edges)
    ancestor_cache: dict[str, dict[str, int]] = {}
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    domain_nodes: dict[tuple[str, str], list[str]] = defaultdict(list)
    for synset_id, node in graph.nodes.items():
        domain_nodes[(node.pos, node.lexname)].append(synset_id)
    for child, parent in direct:
        children_by_parent[parent].append(child)

    for child, parent in direct:
        yield pair_row(graph, child, parent, "direct", split, 1)
    for child, parent in direct:
        yield pair_row(graph, parent, child, "reverse", split, -1)

    for child in sorted(graph.nodes):
        ancestors = graph.ancestors(child, max_distance=64)
        ancestor_cache[child] = ancestors
        candidates = [(node, distance) for node, distance in ancestors.items() if 2 <= distance <= 4]
        rng.shuffle(candidates)
        for parent, distance in candidates[:2]:
            yield pair_row(graph, child, parent, "ancestor", split, distance)

    emitted_sibling: set[tuple[str, str]] = set()
    sibling_lookup: dict[str, set[str]] = defaultdict(set)
    for parent, children in children_by_parent.items():
        ordered = sorted(set(children))
        for child in ordered:
            alternatives = [other for other in ordered if other != child]
            sibling_lookup[child].update(alternatives)
            rng.shuffle(alternatives)
            taken = 0
            for other in alternatives:
                if other in ancestor_cache.get(child, {}) or child in ancestor_cache.get(other, {}):
                    continue
                key = (child, other)
                if key in emitted_sibling:
                    continue
                emitted_sibling.add(key)
                sibling_lookup[child].add(other)
                yield pair_row(graph, child, other, "sibling", split)
                taken += 1
                if taken == 2:
                    break

    for child in sorted(graph.nodes):
        node = graph.nodes[child]
        candidates = domain_nodes[(node.pos, node.lexname)]
        if len(candidates) < 2:
            continue
        forbidden = set(ancestor_cache.get(child, {})) | set(graph.parents.get(child, ()))
        forbidden |= sibling_lookup.get(child, set()) | {child}
        # Reject reverse ancestry too. Try a bounded number of same-domain
        # draws so unrelated examples remain deterministic and inexpensive.
        start = rng.randrange(len(candidates))
        for offset in range(min(64, len(candidates))):
            other = candidates[(start + offset) % len(candidates)]
            if other in forbidden or child in ancestor_cache.get(other, {}):
                continue
            yield pair_row(graph, child, other, "unrelated", split)
            break


def root_rows(graph: WordNetGraph, split: str):
    for synset_id in sorted(graph.nodes):
        node = graph.nodes[synset_id]
        parents = sorted(graph.parents.get(synset_id, ()))
        yield {
            "example_id": stable_id(split, "root", synset_id),
            "split": split,
            "child_synset": synset_id,
            "child": node.surface,
            "child_gloss": node.gloss,
            "lexname": node.lexname,
            "is_root": int(not parents),
            "available_parents": parents,
        }


def sample_subgraphs(graph: WordNetGraph, split: str, seed: int, size: int, count: int):
    neighbors: dict[str, set[str]] = defaultdict(set)
    for child, parent in graph.edges:
        neighbors[child].add(parent)
        neighbors[parent].add(child)
    eligible = [node for node in sorted(graph.nodes) if neighbors[node]]
    if not eligible:
        return
    rng = random.Random(seed)
    rng.shuffle(eligible)
    for index in range(count):
        start = eligible[index % len(eligible)]
        selected: list[str] = []
        seen = {start}
        queue = deque([start])
        while queue and len(selected) < size:
            node = queue.popleft()
            selected.append(node)
            adjacent = sorted(neighbors[node])
            rng.shuffle(adjacent)
            for other in adjacent:
                if other not in seen:
                    seen.add(other)
                    queue.append(other)
        if len(selected) < min(8, size):
            continue
        node_set = set(selected)
        edges = sorted(
            (child, parent) for child, parent in graph.edges
            if child in node_set and parent in node_set
        )
        roots = sorted(
            node for node in selected
            if not any(parent in node_set for parent in graph.parents.get(node, ()))
        )
        yield {
            "subgraph_id": f"{split}-{index:05d}",
            "split": split,
            "nodes": selected,
            "edges": edges,
            "roots": roots,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wordnet-dict", type=Path, default=DEFAULT_DICT)
    parser.add_argument("--bench-root", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subgraph-size", type=int, default=64)
    parser.add_argument("--subgraphs-per-split", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_graph = load_wordnet_graph(args.wordnet_dict)
    blacklist, blacklist_counts = load_blacklist(args.bench_root)
    graph, removed = clean_graph(raw_graph, blacklist)
    domain_split, by_split = assign_domains(graph, args.seed)

    relation_counts: dict[str, dict[str, int]] = {}
    root_counts: dict[str, dict[str, int]] = {}
    split_synsets: dict[str, set[str]] = {}
    for offset, split in enumerate(("train", "dev", "test")):
        keep = set(by_split[split])
        split_synsets[split] = keep
        split_graph = graph.induced(keep)
        counts = Counter()

        def counted_rows():
            for row in relation_rows(split_graph, split, args.seed + offset):
                counts[row["relation"]] += 1
                yield row

        write_jsonl(args.out_root / "relations" / f"{split}.jsonl", counted_rows())
        relation_counts[split] = dict(counts)
        roots = Counter()

        def counted_roots():
            for row in root_rows(split_graph, split):
                roots[str(row["is_root"])] += 1
                yield row

        write_jsonl(args.out_root / "roots" / f"{split}.jsonl", counted_roots())
        root_counts[split] = dict(roots)
        write_jsonl(
            args.out_root / "subgraphs" / f"{split}.jsonl",
            sample_subgraphs(
                split_graph, split, args.seed + 100 + offset,
                args.subgraph_size, args.subgraphs_per_split,
            ),
        )

    split_overlap = {
        "train_dev": len(split_synsets["train"] & split_synsets["dev"]),
        "train_test": len(split_synsets["train"] & split_synsets["test"]),
        "dev_test": len(split_synsets["dev"] & split_synsets["test"]),
    }
    remaining_external_overlap = sum(
        1 for node in graph.nodes.values()
        if any(normalize(lemma) in blacklist for lemma in node.lemmas)
    )
    audit = {
        "schema_version": 1,
        "seed": args.seed,
        "raw_synsets": len(raw_graph.nodes),
        "raw_direct_edges": len(raw_graph.edges),
        "blacklist_vocab": blacklist_counts,
        "removed_external_synsets": len(removed),
        "clean_synsets": len(graph.nodes),
        "clean_direct_edges": len(graph.edges),
        "domain_split": domain_split,
        "split_synsets": {key: len(value) for key, value in split_synsets.items()},
        "relation_counts": relation_counts,
        "root_counts": root_counts,
        "acceptance": {
            "split_synset_overlap": split_overlap,
            "remaining_external_synset_overlap": remaining_external_overlap,
            "passed": not any(split_overlap.values()) and remaining_external_overlap == 0,
        },
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not audit["acceptance"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
