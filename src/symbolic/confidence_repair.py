"""Confidence-aware minimal symbolic repair over shared neural edge scores.

The module deliberately consumes a frozen scored-candidate payload.  Every
variant therefore sees identical neural evidence and differs only in the
explicit symbolic policy applied to that evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable

import networkx as nx

from src.symbolic.maximum_branching import ROOT, BranchCandidate, maximum_branching


@dataclass(frozen=True)
class RelationWeights:
    ancestor: float = 0.5
    sibling: float = 0.25
    reverse: float = 1.0
    unrelated: float = 0.25


@dataclass(frozen=True)
class SymbolicPolicy:
    """Dev-tunable policy; no target-domain labels are consumed."""

    edit_base: float = 0.10
    edit_scale: float = 0.50
    max_edit_penalty: float = 2.0
    max_root_fraction: float = 0.10
    max_roots: int = 0
    root_penalty: float = 0.0
    multi_parent_limit: int = 2
    multi_parent_margin: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict | None) -> "SymbolicPolicy":
        payload = payload or {}
        if "policy" in payload:
            payload = payload["policy"]
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


def _relations(item: dict) -> dict[str, float] | None:
    values = item.get("relations")
    return values if values and "direct" in values else None


def edge_score(
    item: dict,
    stage: str = "relation",
    weights: RelationWeights | None = None,
) -> float:
    if stage == "embedding":
        return float(item.get("retrieval_score", 0.0))
    if stage not in {"relation", "context"}:
        raise ValueError(f"unknown score stage: {stage}")
    probabilities = _relations(item)
    if weights is not None and probabilities is not None:
        score = math.log(max(float(probabilities["direct"]), 1e-12))
        score -= weights.ancestor * float(probabilities.get("ancestor", 0.0))
        score -= weights.sibling * float(probabilities.get("sibling", 0.0))
        score -= weights.reverse * float(probabilities.get("reverse", 0.0))
        score -= weights.unrelated * float(probabilities.get("unrelated", 0.0))
    else:
        score = float(item.get("local_score", item.get("edge_score", 0.0)))
    if stage == "context":
        score += float(item.get("context_delta", 0.0))
    return score


def root_score(row: dict, *, learned: bool = True, root_penalty: float = 0.0) -> float:
    return (float(row.get("root_logit", 0.0)) if learned else 0.0) - root_penalty


def _nodes(payload: dict) -> set[str]:
    return {str(row["child"]) for row in payload.get("proposals", [])}


def _choices(
    payload: dict,
    *,
    stage: str,
    weights: RelationWeights | None,
    learned_root: bool,
    root_penalty: float,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, float]]:
    nodes = _nodes(payload)
    choices: dict[str, list[tuple[str, float]]] = {}
    roots = {}
    for row in payload.get("proposals", []):
        child = str(row["child"])
        best_by_parent: dict[str, float] = {}
        for item in row.get("candidates", []):
            parent = str(item["parent"])
            if parent == child or parent not in nodes:
                continue
            score = edge_score(item, stage, weights)
            best_by_parent[parent] = max(best_by_parent.get(parent, -math.inf), score)
        choices[child] = sorted(best_by_parent.items(), key=lambda pair: (-pair[1], pair[0]))
        roots[child] = root_score(row, learned=learned_root, root_penalty=root_penalty)
    return choices, roots


def independent_parents(
    choices: dict[str, list[tuple[str, float]]], roots: dict[str, float], *, compare_root: bool
) -> tuple[dict[str, str], dict[str, float], dict[str, float]]:
    parents: dict[str, str] = {}
    selected: dict[str, float] = {}
    margins: dict[str, float] = {}
    for child, candidates in choices.items():
        ranked = list(candidates)
        if compare_root or not ranked:
            ranked.append((ROOT, roots[child]))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        parents[child], selected[child] = ranked[0]
        margins[child] = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else math.inf
    return parents, selected, margins


def _allowed_roots(
    roots: dict[str, float], policy: SymbolicPolicy,
    choices: dict[str, list[tuple[str, float]]],
) -> set[str] | None:
    if policy.max_roots <= 0 and policy.max_root_fraction >= 1.0:
        return None
    fraction_cap = max(1, math.ceil(len(roots) * max(0.0, policy.max_root_fraction)))
    cap = policy.max_roots if policy.max_roots > 0 else fraction_cap
    if policy.max_roots > 0 and policy.max_root_fraction < 1.0:
        cap = min(cap, fraction_cap)
    candidate_graph = nx.DiGraph()
    candidate_graph.add_nodes_from(choices)
    candidate_graph.add_edges_from(
        (parent, child) for child, rows in choices.items() for parent, _ in rows
    )
    condensation = nx.condensation(candidate_graph)
    mandatory = set()
    for component in condensation.nodes:
        if condensation.in_degree(component) == 0:
            members = condensation.nodes[component]["members"]
            mandatory.add(max(members, key=lambda node: (roots[node], node)))
    target = max(cap, len(mandatory))
    ranked = sorted(roots, key=lambda child: (-roots[child], child))
    return mandatory | set([child for child in ranked if child not in mandatory][:target-len(mandatory)])


def _graph_stats(parents: dict[str, str | None], extra_edges: Iterable[tuple[str, str]] = ()) -> dict:
    edges = [(child, parent) for child, parent in parents.items() if parent not in {None, ROOT}]
    edges.extend(extra_edges)
    graph = nx.DiGraph()
    graph.add_nodes_from(parents)
    graph.add_edges_from(edges)
    parent_counts = dict(graph.out_degree())
    cyclic_components = [
        sorted(component)
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or any(graph.has_edge(node, node) for node in component)
    ]
    roots = [child for child, parent in parents.items() if parent in {None, ROOT}]
    return {
        "is_dag": not cyclic_components,
        "single_parent": all(count <= 1 for count in parent_counts.values()),
        "valid": not cyclic_components and all(count <= 1 for count in parent_counts.values()),
        "n_cycles_scc": len(cyclic_components),
        "n_cyclic_nodes": sum(map(len, cyclic_components)),
        "cyclic_nodes": sorted({node for component in cyclic_components for node in component}),
        "n_roots": len(roots),
        "n_nodes": len(parents),
        "n_edges": len(edges),
    }


def _variant(
    name: str,
    parents: dict[str, str | None],
    scores: dict[str, float],
    *,
    reference: dict[str, str] | None = None,
    margins: dict[str, float] | None = None,
    extra_edges: Iterable[tuple[str, str]] = (),
    metadata: dict | None = None,
) -> dict:
    extra_edges = list(extra_edges)
    edges = [(child, parent) for child, parent in parents.items() if parent not in {None, ROOT}]
    edges.extend(extra_edges)
    roots = sorted(child for child, parent in parents.items() if parent in {None, ROOT})
    changed = sorted(child for child in parents if reference is not None and parents[child] != reference.get(child))
    return {
        "system": name,
        "edges": [
            {"child": child, "parent": parent, "score": float(scores.get(child, 0.0))}
            for child, parent in sorted(set(edges))
        ],
        "roots": roots,
        "graph_stats": _graph_stats(parents, extra_edges),
        "repair_certificate": {
            "reference": "relation_root_top1" if reference is not None else None,
            "n_changed": len(changed),
            "changed_rate": len(changed) / max(1, len(parents)),
            "changed_children": changed,
            "confidence_margins": {child: float(margins[child]) for child in changed} if margins else {},
        },
        "policy": metadata or {},
    }


def exact_parents(
    choices: dict[str, list[tuple[str, float]]],
    roots: dict[str, float],
    *,
    policy: SymbolicPolicy,
    reference: dict[str, str] | None = None,
    margins: dict[str, float] | None = None,
    use_edit_penalty: bool = False,
    use_root_budget: bool = False,
) -> tuple[dict[str, str], dict[str, float], dict]:
    adjusted_roots = dict(roots)
    candidates: list[BranchCandidate] = []
    edit_penalties = {}
    for child, rows in choices.items():
        margin = 0.0 if margins is None else max(0.0, min(float(margins[child]), policy.max_edit_penalty))
        penalty = min(policy.max_edit_penalty, policy.edit_base + policy.edit_scale * margin)
        edit_penalties[child] = penalty if use_edit_penalty else 0.0
        preferred = reference.get(child) if reference else None
        for parent, score in rows:
            bonus = penalty if use_edit_penalty and parent == preferred else 0.0
            candidates.append(BranchCandidate(child, parent, score + bonus))
        if use_edit_penalty and preferred == ROOT:
            adjusted_roots[child] += penalty
    allowed = _allowed_roots(adjusted_roots, policy, choices) if use_root_budget else None
    result = maximum_branching(set(choices), candidates, adjusted_roots, allowed_roots=allowed)
    score_lookup = {(candidate.child, candidate.parent): candidate.score for candidate in candidates}
    scores = {
        child: adjusted_roots[child] if parent == ROOT else score_lookup[(child, parent)]
        for child, parent in result.parents.items()
    }
    meta = {
        "objective": "maximum_branching",
        "acyclicity": True,
        "single_parent_or_root": True,
        "confidence_edit_penalty": use_edit_penalty,
        "root_budget": use_root_budget,
        "allowed_root_count": len(allowed) if allowed is not None else len(choices),
        "edit_penalties": edit_penalties,
        "optimal_score": result.score,
    }
    return result.parents, scores, meta


def _multi_parent_variant(
    choices: dict[str, list[tuple[str, float]]], roots: dict[str, float], policy: SymbolicPolicy
) -> tuple[dict[str, str], dict[str, float], list[tuple[str, str]]]:
    parents: dict[str, str] = {}
    scores: dict[str, float] = {}
    extras: list[tuple[str, str]] = []
    for child, ranked in choices.items():
        if not ranked or roots[child] >= ranked[0][1]:
            parents[child], scores[child] = ROOT, roots[child]
            continue
        parents[child], scores[child] = ranked[0]
        cutoff = max(roots[child], ranked[0][1] - policy.multi_parent_margin)
        extras.extend((child, parent) for parent, score in ranked[1:policy.multi_parent_limit] if score >= cutoff)
    return parents, scores, extras


def decode_neurosymbolic(payload: dict, policy: SymbolicPolicy | None = None) -> dict:
    policy = policy or SymbolicPolicy()
    typed = RelationWeights()
    relation, learned_roots = _choices(
        payload, stage="relation", weights=None, learned_root=True, root_penalty=policy.root_penalty
    )
    context, context_roots = _choices(
        payload, stage="context", weights=None, learned_root=True, root_penalty=policy.root_penalty
    )
    embedding, _ = _choices(payload, stage="embedding", weights=None, learned_root=False, root_penalty=0.0)
    reference, reference_scores, margins = independent_parents(relation, learned_roots, compare_root=True)
    variants = {}

    emb_parent, emb_scores, _ = independent_parents(embedding, learned_roots, compare_root=False)
    variants["embedding_top1"] = _variant("embedding_top1", emb_parent, emb_scores)
    rel_parent, rel_scores, _ = independent_parents(relation, learned_roots, compare_root=False)
    variants["relation_top1"] = _variant("relation_top1", rel_parent, rel_scores)
    variants["relation_root_top1"] = _variant("relation_root_top1", reference, reference_scores)

    exact, scores, meta = exact_parents(relation, learned_roots, policy=policy)
    variants["relation_root_exact"] = _variant(
        "relation_root_exact", exact, scores, reference=reference, margins=margins, metadata=meta
    )
    minimal, scores, meta = exact_parents(
        relation, learned_roots, policy=policy, reference=reference, margins=margins, use_edit_penalty=True
    )
    variants["relation_root_minimal_edit"] = _variant(
        "relation_root_minimal_edit", minimal, scores, reference=reference, margins=margins, metadata=meta
    )
    budget, scores, meta = exact_parents(
        relation, learned_roots, policy=policy, use_root_budget=True
    )
    variants["relation_root_budget_exact"] = _variant(
        "relation_root_budget_exact", budget, scores, reference=reference, margins=margins, metadata=meta
    )
    confidence, scores, meta = exact_parents(
        relation, learned_roots, policy=policy, reference=reference, margins=margins,
        use_edit_penalty=True, use_root_budget=True,
    )
    variants["confidence_symbolic"] = _variant(
        "confidence_symbolic", confidence, scores, reference=reference, margins=margins, metadata=meta
    )

    context_reference, context_scores, _ = independent_parents(context, context_roots, compare_root=True)
    variants["context_top1"] = _variant("context_top1", context_reference, context_scores)
    context_exact, scores, meta = exact_parents(context, context_roots, policy=SymbolicPolicy(max_root_fraction=1.0))
    variants["exact_branching"] = _variant(
        "exact_branching", context_exact, scores, reference=context_reference, metadata=meta
    )

    variants["ablation_no_acyclicity"] = _variant(
        "ablation_no_acyclicity", reference, reference_scores, metadata={"acyclicity": False}
    )
    multi, multi_scores, extras = _multi_parent_variant(relation, learned_roots, policy)
    variants["ablation_no_single_parent"] = _variant(
        "ablation_no_single_parent", multi, multi_scores, extra_edges=extras,
        metadata={"single_parent_or_root": False, "selection": "top-k-within-margin"},
    )
    _, uniform_roots = _choices(
        payload, stage="relation", weights=None, learned_root=False, root_penalty=0.0
    )
    no_root, scores, meta = exact_parents(relation, uniform_roots, policy=policy, use_root_budget=True)
    variants["ablation_no_root_scores"] = _variant(
        "ablation_no_root_scores", no_root, scores, reference=reference, margins=margins, metadata=meta
    )

    weight_ablations = {
        "ablation_no_relation_penalties": RelationWeights(0.0, 0.0, 0.0, 0.0),
        "ablation_no_ancestor_penalty": RelationWeights(0.0, typed.sibling, typed.reverse, typed.unrelated),
        "ablation_no_sibling_penalty": RelationWeights(typed.ancestor, 0.0, typed.reverse, typed.unrelated),
        "ablation_no_reverse_penalty": RelationWeights(typed.ancestor, typed.sibling, 0.0, typed.unrelated),
        "ablation_no_unrelated_penalty": RelationWeights(typed.ancestor, typed.sibling, typed.reverse, 0.0),
    }
    typed_keys = {"direct", "ancestor", "sibling", "reverse", "unrelated"}
    relation_ablation_available = any(
        typed_keys <= set(_relations(item) or {})
        for row in payload.get("proposals", []) for item in row.get("candidates", [])
    )
    for name, weights in (weight_ablations.items() if relation_ablation_available else ()):
        scored, roots = _choices(
            payload, stage="relation", weights=weights, learned_root=True, root_penalty=policy.root_penalty
        )
        parents, _, ablation_margins = independent_parents(scored, roots, compare_root=True)
        decoded, decoded_scores, meta = exact_parents(
            scored, roots, policy=policy, reference=parents, margins=ablation_margins,
            use_edit_penalty=True, use_root_budget=True,
        )
        meta["relation_weights"] = asdict(weights)
        variants[name] = _variant(name, decoded, decoded_scores, reference=parents, margins=ablation_margins, metadata=meta)
    variants["ablation_no_confidence_penalty"] = dict(variants["relation_root_budget_exact"], system="ablation_no_confidence_penalty")

    return {
        "schema_version": "taxobranch.neurosymbolic.1",
        "dataset": payload.get("dataset") or payload.get("domain"),
        "shared_candidate_set": True,
        "neural_scores_frozen": True,
        "policy": asdict(policy),
        "variants": variants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.scored.read_text(encoding="utf-8"))
    policy_payload = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else None
    result = decode_neurosymbolic(payload, SymbolicPolicy.from_dict(policy_payload))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "systems": len(result["variants"])}, sort_keys=True))


if __name__ == "__main__":
    main()
