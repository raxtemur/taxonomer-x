"""Exact ROOT-aware maximum branching for single-parent taxonomy induction."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


ROOT = "__ROOT__"


@dataclass(frozen=True)
class BranchCandidate:
    child: str
    parent: str
    score: float


@dataclass(frozen=True)
class FeasibilityCertificate:
    """Independently checkable structural properties of a parent map."""

    covers_all_nodes: bool
    one_choice_per_node: bool
    parents_in_vocabulary: bool
    acyclic: bool

    @property
    def valid(self) -> bool:
        return (
            self.covers_all_nodes
            and self.one_choice_per_node
            and self.parents_in_vocabulary
            and self.acyclic
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "valid": self.valid,
            "covers_all_nodes": self.covers_all_nodes,
            "one_choice_per_node": self.one_choice_per_node,
            "parents_in_vocabulary": self.parents_in_vocabulary,
            "acyclic": self.acyclic,
        }


def verify_feasibility(
    nodes: list[str] | set[str], parents: dict[str, str]
) -> FeasibilityCertificate:
    """Verify coverage, parent validity, total choice, and acyclicity.

    This verifier uses only the declared vocabulary and returned parent map; it
    does not trust decoder internals or make a semantic-correctness claim.
    """
    vocabulary = set(nodes)
    parent_keys = set(parents)
    covers_all_nodes = vocabulary.issubset(parent_keys)
    one_choice_per_node = parent_keys == vocabulary
    parents_in_vocabulary = all(
        child in vocabulary
        and parent != child
        and (parent == ROOT or parent in vocabulary)
        for child, parent in parents.items()
    )
    graph = nx.DiGraph()
    graph.add_nodes_from(vocabulary)
    graph.add_edges_from(
        (child, parent)
        for child, parent in parents.items()
        if child in vocabulary and parent in vocabulary and child != parent
    )
    return FeasibilityCertificate(
        covers_all_nodes=covers_all_nodes,
        one_choice_per_node=one_choice_per_node,
        parents_in_vocabulary=parents_in_vocabulary,
        acyclic=nx.is_directed_acyclic_graph(graph),
    )


@dataclass(frozen=True)
class BranchingResult:
    parents: dict[str, str]
    score: float
    graph: nx.DiGraph

    @property
    def taxonomy_edges(self) -> list[tuple[str, str]]:
        return sorted((child, parent) for child, parent in self.parents.items() if parent != ROOT)

    @property
    def roots(self) -> list[str]:
        return sorted(child for child, parent in self.parents.items() if parent == ROOT)

    @property
    def feasibility_certificate(self) -> FeasibilityCertificate:
        return verify_feasibility(set(self.graph.nodes), self.parents)


def _stable_maximum_spanning_arborescence(graph: nx.DiGraph) -> nx.DiGraph:
    """Exact spanning branching robust to nested negative-weight contractions.

    NetworkX 3.6.1 shifts weights by one global range before calling its
    maximum-branching implementation.  Nested cycle contractions can consume
    that shift and incorrectly return a non-spanning branching.  A
    lexicographic cardinality bonus larger than every possible total-score
    difference first maximizes edge count; all spanning branchings then have
    equal cardinality, so their original-score ordering is unchanged.
    """
    if not graph:
        return graph.copy()
    weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    if not weights:
        raise nx.NetworkXException("No spanning arborescence in an edgeless graph")
    score_bound = max(abs(weight) for weight in weights)
    cardinality_bonus = (2 * len(graph) + 1) * (score_bound + 1.0)
    work = graph.copy()
    attribute = "_cardinality_first_weight"
    for _, _, data in work.edges(data=True):
        data[attribute] = float(data.get("weight", 0.0)) + cardinality_bonus
    optimum = nx.maximum_branching(
        work, attr=attribute, default=cardinality_bonus, preserve_attrs=True
    )
    if not nx.is_arborescence(optimum):
        roots = [node for node, degree in optimum.in_degree() if degree == 0]
        raise nx.NetworkXException(
            f"No spanning arborescence after cardinality-first Edmonds; roots={roots[:5]}"
        )
    for _, _, data in optimum.edges(data=True):
        data.pop(attribute, None)
    return optimum


def maximum_branching(
    nodes: list[str] | set[str],
    candidates: list[BranchCandidate],
    root_scores: dict[str, float] | None = None,
    *,
    allowed_roots: set[str] | None = None,
) -> BranchingResult:
    """Return the exact maximum-score acyclic one-parent-or-ROOT structure.

    Public edge convention is child -> parent. NetworkX arborescences use
    parent -> child, so orientation is reversed only inside this function.
    """
    vocabulary = set(nodes)
    root_scores = root_scores or {}
    if allowed_roots is not None:
        allowed_roots = vocabulary & set(allowed_roots)
        if not allowed_roots:
            raise ValueError("allowed_roots must contain at least one vocabulary node")
    graph = nx.DiGraph()
    graph.add_node(ROOT)
    graph.add_nodes_from(vocabulary)
    for child in vocabulary:
        if allowed_roots is None or child in allowed_roots:
            graph.add_edge(ROOT, child, weight=float(root_scores.get(child, 0.0)), kind="root")
    for candidate in candidates:
        if candidate.child == candidate.parent:
            continue
        if candidate.child not in vocabulary or candidate.parent not in vocabulary:
            continue
        existing = graph.get_edge_data(candidate.parent, candidate.child)
        if existing is None or candidate.score > existing["weight"]:
            graph.add_edge(
                candidate.parent, candidate.child,
                weight=float(candidate.score), kind="taxonomy",
            )
    optimum = _stable_maximum_spanning_arborescence(graph)
    parents: dict[str, str] = {}
    total = 0.0
    for parent, child, data in optimum.edges(data=True):
        if child == ROOT:
            raise AssertionError("Virtual ROOT must not have a parent")
        parents[child] = parent
        total += float(data.get("weight", 0.0))
    if set(parents) != vocabulary:
        missing = sorted(vocabulary - set(parents))
        raise AssertionError(f"Branching omitted nodes: {missing[:5]}")
    taxonomy = nx.DiGraph()
    taxonomy.add_nodes_from(vocabulary)
    taxonomy.add_edges_from((child, parent) for child, parent in parents.items() if parent != ROOT)
    if not nx.is_directed_acyclic_graph(taxonomy):
        raise AssertionError("Maximum branching orientation produced a cycle")
    return BranchingResult(parents=parents, score=total, graph=taxonomy)


def load_scored_candidates(path: Path) -> tuple[set[str], list[BranchCandidate], dict[str, float]]:
    nodes: set[str] = set()
    candidates: list[BranchCandidate] = []
    roots: dict[str, float] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            child = row["child"]
            nodes.add(child)
            roots[child] = float(row.get("root_score", 0.0))
            for item in row.get("candidates", []):
                parent = item["parent"]
                nodes.add(parent)
                candidates.append(BranchCandidate(child, parent, float(item["score"])))
    return nodes, candidates, roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    nodes, candidates, roots = load_scored_candidates(args.scores)
    result = maximum_branching(nodes, candidates, roots)
    payload = {
        "decoder": "exact_maximum_branching",
        "score": result.score,
        "feasibility_certificate": result.feasibility_certificate.as_dict(),
        "roots": result.roots,
        "edges": [{"child": child, "parent": parent} for child, parent in result.taxonomy_edges],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
