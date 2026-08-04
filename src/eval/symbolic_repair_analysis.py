"""Measure what symbolic inference changes, fixes, harms, and cannot recover."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.data.texeval_loader import gold_edges_as_set, load_terms
from src.eval.paired_bootstrap import paired_ancestor_bootstrap, paired_bootstrap
from src.symbolic.confidence_repair import ROOT, edge_score


def _parent_map(variant: dict, universe: set[str]) -> dict[str, str]:
    parents = {child: ROOT for child in universe}
    parents.update({row["child"]: row["parent"] for row in variant.get("edges", [])})
    return parents


def _gold_map(gold: set[tuple[str, str]], universe: set[str]) -> dict[str, set[str]]:
    result = {child: set() for child in universe}
    for child, parent in gold:
        result.setdefault(child, set()).add(parent)
    return result


def _correct(parent: str, gold_parents: set[str]) -> bool:
    return parent == ROOT if not gold_parents else parent in gold_parents


def _cyclic_nodes(parents: dict[str, str]) -> set[str]:
    graph = nx.DiGraph((child, parent) for child, parent in parents.items() if parent != ROOT)
    return {
        node
        for component in nx.strongly_connected_components(graph)
        if len(component) > 1 or any(graph.has_edge(item, item) for item in component)
        for node in component
    }


def _slice(records: list[dict]) -> dict:
    n = len(records)
    baseline = sum(row["baseline_correct"] for row in records)
    final = sum(row["final_correct"] for row in records)
    corrected = sum(row["corrected"] for row in records)
    harmed = sum(row["harmed"] for row in records)
    return {
        "n": n,
        "baseline_accuracy": baseline / max(1, n),
        "final_accuracy": final / max(1, n),
        "accuracy_delta": (final - baseline) / max(1, n),
        "corrected": corrected,
        "harmed": harmed,
        "net_corrections": corrected - harmed,
    }


def _margin_label(margin: float) -> str:
    if margin < 0.10:
        return "lt_0.10"
    if margin < 0.25:
        return "0.10_0.25"
    if margin < 0.50:
        return "0.25_0.50"
    if margin < 1.0:
        return "0.50_1.00"
    return "ge_1.00"


def analyze_repairs(
    payload: dict,
    gold: set[tuple[str, str]],
    universe: set[str],
    *,
    baseline: str = "relation_root_top1",
    system: str = "confidence_symbolic",
    bootstrap_samples: int = 10000,
    seed: int = 42,
) -> dict:
    variants = payload["variants"]
    before = _parent_map(variants[baseline], universe)
    after = _parent_map(variants[system], universe)
    gold_parents = _gold_map(gold, universe)
    cyclic = _cyclic_nodes(before)

    proposals = {row["child"]: row for row in payload.get("source_proposals", [])}
    if not proposals:
        proposals = {row["child"]: row for row in payload.get("proposals", [])}
    candidate_sets = {
        child: {item["parent"] for item in row.get("candidates", [])}
        for child, row in proposals.items()
    }
    reciprocal = {
        child
        for child, candidates in candidate_sets.items()
        if any(child in candidate_sets.get(parent, set()) for parent in candidates)
    }

    records = []
    for child in sorted(universe):
        row = proposals.get(child, {})
        candidates = row.get("candidates", [])
        scored = sorted(
            [(item["parent"], edge_score(item, "relation")) for item in candidates],
            key=lambda pair: (-pair[1], pair[0]),
        )
        root = float(row.get("root_logit", 0.0))
        choices = scored + [(ROOT, root)]
        choices.sort(key=lambda pair: (-pair[1], pair[0]))
        margin = choices[0][1] - choices[1][1] if len(choices) > 1 else math.inf
        covered = not gold_parents[child] or bool(gold_parents[child] & candidate_sets.get(child, set()))
        b_ok = _correct(before[child], gold_parents[child])
        a_ok = _correct(after[child], gold_parents[child])
        changed = before[child] != after[child]
        reasons = []
        if child in cyclic:
            reasons.append("cycle")
        if before[child] == ROOT or after[child] == ROOT or (scored and abs(scored[0][1] - root) < 0.25):
            reasons.append("root_ambiguity")
        if child in reciprocal:
            reasons.append("reciprocal_candidates")
        if changed and not reasons:
            reasons.append("global_objective")
        records.append({
            "child": child,
            "baseline_parent": before[child],
            "final_parent": after[child],
            "gold_parents": sorted(gold_parents[child]),
            "gold_candidate_available": covered,
            "baseline_correct": b_ok,
            "final_correct": a_ok,
            "changed": changed,
            "corrected": changed and not b_ok and a_ok,
            "harmed": changed and b_ok and not a_ok,
            "margin": float(margin),
            "margin_bin": _margin_label(margin),
            "cycle_active": child in cyclic,
            "reciprocal_active": child in reciprocal,
            "root_ambiguous": "root_ambiguity" in reasons,
            "reasons": reasons,
        })

    changed = [row for row in records if row["changed"]]
    corrected = [row for row in records if row["corrected"]]
    harmed = [row for row in records if row["harmed"]]
    pred_before = {(child, parent) for child, parent in before.items() if parent != ROOT}
    pred_after = {(child, parent) for child, parent in after.items() if parent != ROOT}
    graph_before = nx.DiGraph(pred_before); graph_before.add_nodes_from(universe)
    graph_after = nx.DiGraph(pred_after); graph_after.add_nodes_from(universe)
    graph_gold = nx.DiGraph(gold); graph_gold.add_nodes_from(universe)
    ancestors_before = {node: nx.descendants(graph_before, node) for node in universe}
    ancestors_after = {node: nx.descendants(graph_after, node) for node in universe}
    ancestors_gold = {node: nx.descendants(graph_gold, node) for node in universe}
    score_lookup = {}
    for child, proposal in proposals.items():
        values = {item["parent"]: edge_score(item, "relation") for item in proposal.get("candidates", [])}
        values[ROOT] = float(proposal.get("root_logit", 0.0))
        score_lookup[child] = values
    for record in records:
        child = record["child"]
        affected = {child} | nx.ancestors(graph_before, child) | nx.ancestors(graph_after, child)
        baseline_score = score_lookup.get(child, {}).get(record["baseline_parent"], float("nan"))
        final_score = score_lookup.get(child, {}).get(record["final_parent"], float("nan"))
        record.update({
            "baseline_score": float(baseline_score),
            "final_score": float(final_score),
            "score_sacrifice": float(baseline_score - final_score),
            "affected_subtree_nodes": len(affected),
            "observed_subtree_ancestor_flips": sum(len(ancestors_before[node] ^ ancestors_after[node]) for node in affected),
            "observed_subtree_gold_gains": sum(len((ancestors_after[node] - ancestors_before[node]) & ancestors_gold[node]) for node in affected),
            "observed_subtree_gold_losses": sum(len((ancestors_before[node] - ancestors_after[node]) & ancestors_gold[node]) for node in affected),
        })
    before_pairs = {(node, ancestor) for node, values in ancestors_before.items() for ancestor in values}
    after_pairs = {(node, ancestor) for node, values in ancestors_after.items() for ancestor in values}
    gold_pairs = {(node, ancestor) for node, values in ancestors_gold.items() for ancestor in values}
    groups = {
        "constraint_active": changed,
        "constraint_inactive": [row for row in records if not row["changed"]],
        "cycle_nodes": [row for row in records if row["cycle_active"]],
        "non_cycle_nodes": [row for row in records if not row["cycle_active"]],
        "reciprocal_candidates": [row for row in records if row["reciprocal_active"]],
        "root_ambiguous": [row for row in records if row["root_ambiguous"]],
        "gold_candidate_available": [row for row in records if row["gold_candidate_available"]],
        "retrieval_miss": [row for row in records if not row["gold_candidate_available"]],
    }
    for label in sorted({row["margin_bin"] for row in records}):
        groups[f"margin.{label}"] = [row for row in records if row["margin_bin"] == label]

    retrieval_failures = sum(not row["gold_candidate_available"] and not row["final_correct"] for row in records)
    neural_failures = sum(row["gold_candidate_available"] and not row["baseline_correct"] for row in records)
    unresolved_neural = sum(
        row["gold_candidate_available"] and not row["baseline_correct"] and not row["final_correct"]
        for row in records
    )
    return {
        "schema_version": "taxobranch.symbolic_repair_analysis.2",
        "dataset": payload.get("dataset"),
        "baseline": baseline,
        "system": system,
        "n_nodes": len(records),
        "repair": {
            "changed": len(changed),
            "changed_rate": len(changed) / max(1, len(records)),
            "corrected": len(corrected),
            "harmed": len(harmed),
            "neutral_changes": len(changed) - len(corrected) - len(harmed),
            "correction_precision": len(corrected) / max(1, len(changed)),
            "harm_rate": len(harmed) / max(1, len(changed)),
            "net_corrections": len(corrected) - len(harmed),
            "changed_trace": changed,
        },
        "violations": {
            "before": variants[baseline].get("graph_stats", {}),
            "after": variants[system].get("graph_stats", {}),
        },
        "failure_attribution": {
            "retrieval_failures": retrieval_failures,
            "neural_scoring_failures_before_inference": neural_failures,
            "neural_failures_unresolved_by_symbolic": unresolved_neural,
            "symbolic_corrections": len(corrected),
            "symbolic_harms": len(harmed),
            "denominator": len(records),
        },
        "slices": {name: _slice(rows) for name, rows in groups.items()},
        "ancestor_pair_audit": {
            "before": len(before_pairs), "after": len(after_pairs),
            "added": len(after_pairs - before_pairs), "removed": len(before_pairs - after_pairs),
            "gold_gained": len((after_pairs - before_pairs) & gold_pairs),
            "gold_lost": len((before_pairs - after_pairs) & gold_pairs),
            "net_gold": len(after_pairs & gold_pairs) - len(before_pairs & gold_pairs),
        },
        "paired_bootstrap": paired_bootstrap(
            pred_before, pred_after, gold, bootstrap_samples, seed
        ),
        "paired_bootstrap_edge": paired_bootstrap(
            pred_before, pred_after, gold, bootstrap_samples, seed
        ),
        "paired_bootstrap_ancestor": paired_ancestor_bootstrap(
            pred_before, pred_after, gold, universe, bootstrap_samples, seed
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decoded", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--baseline", default="relation_root_top1")
    parser.add_argument("--system", default="confidence_symbolic")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.decoded.read_text(encoding="utf-8"))
    scored = json.loads(args.scored.read_text(encoding="utf-8"))
    payload["source_proposals"] = scored.get("proposals", [])
    gold = gold_edges_as_set(args.domain)
    universe = set(load_terms(args.domain)) | {node for edge in gold for node in edge}
    result = analyze_repairs(
        payload, gold, universe, baseline=args.baseline, system=args.system,
        bootstrap_samples=args.samples, seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **result["repair"]}, default=str))


if __name__ == "__main__":
    main()
