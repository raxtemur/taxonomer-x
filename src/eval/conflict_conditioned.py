"""Post-allocation audit stratified by pre-allocation structural conflict."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx

from src.symbolic.maximum_branching import BranchingResult, ROOT


def _ancestors(node: str, parents: dict[str, str]) -> set[str]:
    result = set()
    while parents[node] != ROOT:
        node = parents[node]
        if node in result:
            raise ValueError("decoded structure must be acyclic")
        result.add(node)
    return result


def conflict_conditioned_audit(
    nodes: list[str],
    gold: set[tuple[str, str]],
    before: BranchingResult,
    after: BranchingResult,
    features: list[dict[str, float]],
    *,
    low_margin_quantile: float = 0.25,
) -> dict:
    """Use labels only after allocation to describe where changes help or harm."""
    gold_parents: dict[str, set[str]] = defaultdict(set)
    graph = nx.DiGraph(); graph.add_nodes_from(nodes); graph.add_edges_from(gold)
    for child, parent in gold:
        gold_parents[child].add(parent)
    uncertainties = sorted(float(row["exchange_uncertainty"]) for row in features)
    threshold = uncertainties[int((len(uncertainties) - 1) * (1.0 - low_margin_quantile))] if uncertainties else 1.0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(features):
        labels = ["all"]
        active = False
        for label, condition in (
            ("cycle_active", row.get("in_cycle", 0.0) > 0.0),
            ("projection_edited", row.get("projection_edited", 0.0) > 0.0),
            ("root_conflict", row.get("root_conflict", 0.0) > 0.0),
            ("low_exchange_margin", row.get("exchange_uncertainty", 0.0) >= threshold),
            ("amplified_subtree", row.get("subtree_size_log", 0.0) > 1.0),
        ):
            if condition:
                labels.append(label); active = True
        labels.append("any_structural_conflict" if active else "inactive")
        for label in labels:
            groups[label].append(index)

    output = {}
    for name, indices in groups.items():
        changed = corrected = harmed = 0
        edge_before = edge_after = ancestor_delta = 0.0
        for index in indices:
            node = nodes[index]
            gold_for_node = gold_parents.get(node, {ROOT})
            old_ok = before.parents[node] in gold_for_node
            new_ok = after.parents[node] in gold_for_node
            edge_before += float(old_ok); edge_after += float(new_ok)
            changed += int(before.parents[node] != after.parents[node])
            corrected += int(not old_ok and new_ok); harmed += int(old_ok and not new_ok)
            gold_anc = set(nx.descendants(graph, node))
            old_anc, new_anc = _ancestors(node, before.parents), _ancestors(node, after.parents)
            def f1(predicted):
                return 1.0 if not predicted and not gold_anc else (
                    0.0 if not predicted or not gold_anc else 2 * len(predicted & gold_anc) / (len(predicted) + len(gold_anc))
                )
            ancestor_delta += f1(new_anc) - f1(old_anc)
        count = len(indices)
        output[name] = {
            "n": count,
            "changed": changed,
            "corrected": corrected,
            "harmed": harmed,
            "edge_accuracy_before": edge_before / max(1, count),
            "edge_accuracy_after": edge_after / max(1, count),
            "edge_accuracy_delta": (edge_after - edge_before) / max(1, count),
            "mean_node_ancestor_f1_delta": ancestor_delta / max(1, count),
        }
    return {
        "strata": output,
        "low_margin_rule": f"top {low_margin_quantile:.0%} exchange_uncertainty within graph",
        "labels_used_only_for_post_allocation_audit": True,
    }

def intervention_enrichment_audit(nodes, gold, before, after, selected_children, features):
    """Describe repair/harm enrichment inside a pre-label acquisition set."""
    if len(nodes) != len(features): raise ValueError("nodes and features must align")
    gold_parents=defaultdict(set)
    for child,parent in gold: gold_parents[child].add(parent)
    def summarize(indices):
        corrected=harmed=changed=initially_wrong=0; tension=0.0
        for index in indices:
            node=nodes[index]; acceptable=gold_parents.get(node,{ROOT})
            old_ok=before.parents[node] in acceptable; new_ok=after.parents[node] in acceptable
            changed+=int(before.parents[node]!=after.parents[node]); initially_wrong+=int(not old_ok)
            corrected+=int(not old_ok and new_ok); harmed+=int(old_ok and not new_ok)
            row=features[index]; tension+=float(row.get("score_sacrifice",0.0))+float(row.get("in_cycle",0.0))
        n=len(indices)
        return {"n":n,"changed":changed,"corrected":corrected,"harmed":harmed,
                "repair_precision":corrected/max(1,changed),
                "repair_recall_among_initial_errors":corrected/max(1,initially_wrong),
                "net_repairs_per_node":(corrected-harmed)/max(1,n),
                "mean_preallocation_tension":tension/max(1,n)}
    selected=summarize(sorted(selected_children))
    unselected=summarize([i for i in range(len(nodes)) if i not in selected_children])
    return {"selected":selected,"unselected":unselected,
            "repair_precision_enrichment":selected["repair_precision"]-unselected["repair_precision"],
            "net_repair_enrichment":selected["net_repairs_per_node"]-unselected["net_repairs_per_node"],
            "labels_used_only_after_allocation":True,"audit_used_for_selection":False}
