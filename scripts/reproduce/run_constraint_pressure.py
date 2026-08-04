"""Run a frozen synthetic competing-cycle pressure suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from src.symbolic.maximum_branching import ROOT, BranchCandidate, maximum_branching


@dataclass(frozen=True)
class SyntheticInstance:
    nodes: list[str]
    candidates: list[BranchCandidate]
    roots: dict[str, float]
    gold: dict[str, str]
    injected_nodes: set[str]


def _score_lookup(instance: SyntheticInstance) -> dict[tuple[str, str], float]:
    lookup = {(node, ROOT): score for node, score in instance.roots.items()}
    for row in instance.candidates:
        key = (row.child, row.parent)
        lookup[key] = max(lookup.get(key, float("-inf")), row.score)
    return lookup


def generate_instance(size: int, length: int, pressure: float, mode: str, seed: int, distractors: int) -> SyntheticInstance:
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(size)]
    gold = {nodes[0]: ROOT}
    candidates: list[BranchCandidate] = []
    roots = {node: (-3.0 + rng.uniform(-0.05, 0.05)) for node in nodes}
    roots[nodes[0]] = 6.0
    for index in range(1, size):
        child = nodes[index]
        parent = nodes[(index - 1) // 2]
        gold[child] = parent
        candidates.append(BranchCandidate(child, parent, 4.0 + rng.uniform(-0.1, 0.1)))
        pool = [node for node in nodes[:index] if node != parent]
        rng.shuffle(pool)
        for alt in pool[:distractors]:
            candidates.append(BranchCandidate(child, alt, 1.0 + rng.uniform(-0.25, 0.25)))
    eligible = nodes[1:]
    rng.shuffle(eligible)
    target = max(length, int(round(pressure * len(eligible))))
    groups = max(1, target // length)
    injected: set[str] = set()
    for group_index in range(groups):
        start = group_index * length
        group = eligible[start:start + length]
        if len(group) < length:
            break
        injected.update(group)
        for i, child in enumerate(group):
            candidates.append(BranchCandidate(child, group[(i + 1) % length], 8.0 + rng.uniform(-0.08, 0.08)))
        if mode == "overlapping" and length >= 3:
            # A second, nearly tied cycle over the same nodes creates competing
            # global repairs rather than merely independent disjoint cycles.
            for i, child in enumerate(group):
                candidates.append(BranchCandidate(child, group[(i + 2) % length], 7.75 + rng.uniform(-0.08, 0.08)))
            # A selected tail plus an attractive reverse edge creates a genuine
            # edit-vs-objective trade-off: one expensive edit can break the
            # primary cycle, while two cheaper edits traverse the satellite.
            satellite_index = groups * length + group_index
            if satellite_index < len(eligible):
                satellite = eligible[satellite_index]
                injected.add(satellite)
                candidates.append(BranchCandidate(satellite, group[0], 5.4 + rng.uniform(-0.04, 0.04)))
                candidates.append(BranchCandidate(group[0], satellite, 7.88 + rng.uniform(-0.04, 0.04)))
    return SyntheticInstance(nodes, candidates, roots, gold, injected)


def _independent(instance: SyntheticInstance) -> tuple[dict[str, str], float]:
    lookup = _score_lookup(instance)
    choices = {}
    for child in instance.nodes:
        options = [(score, parent) for (node, parent), score in lookup.items() if node == child]
        score, parent = max(options, key=lambda row: (row[0], row[1]))
        choices[child] = parent
    return choices, sum(lookup[(child, parent)] for child, parent in choices.items())


def _graph(nodes: list[str], choices: dict[str, str]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((child, parent) for child, parent in choices.items() if parent != ROOT)
    return graph


def _cycle_components(graph: nx.DiGraph) -> list[set[str]]:
    # Selected-parent graphs are functional, so every non-trivial SCC contains
    # exactly one directed cycle; SCC enumeration is linear rather than the
    # general simple-cycle enumeration used by networkx.simple_cycles.
    return [set(component) for component in nx.strongly_connected_components(graph) if len(component) > 1]


def _greedy(instance: SyntheticInstance, initial: dict[str, str]) -> tuple[dict[str, str], float]:
    lookup = _score_lookup(instance)
    choices = dict(initial)
    while True:
        cycles = _cycle_components(_graph(instance.nodes, choices))
        if not cycles:
            break
        cycle = min(cycles, key=lambda row: (len(row), tuple(sorted(row))))
        repairs = []
        for child in sorted(cycle):
            current = choices[child]
            current_score = lookup[(child, current)]
            options = [(parent, score) for (node, parent), score in lookup.items() if node == child and parent != current]
            for parent, score in options:
                trial = dict(choices)
                trial[child] = parent
                trial_cycles = _cycle_components(_graph(instance.nodes, trial))
                # The changed child must leave every cycle. This prevents a
                # cheap replacement from shrinking the same cycle forever.
                if not any(child in row for row in trial_cycles):
                    repairs.append((current_score - score, child, parent))
        if not repairs:
            raise RuntimeError("ROOT option should always permit a greedy cycle repair")
        _, child, parent = min(repairs)
        choices[child] = parent
    return choices, sum(lookup[(child, parent)] for child, parent in choices.items())


def _minimal_edit(instance: SyntheticInstance, initial: dict[str, str]) -> tuple[dict[str, str], float]:
    lookup = _score_lookup(instance)
    span = sum(max(abs(value), 1.0) for value in lookup.values()) + 1.0
    roots = {node: score + (span if initial[node] == ROOT else 0.0) for node, score in instance.roots.items()}
    candidates = [BranchCandidate(row.child, row.parent, row.score + (span if initial[row.child] == row.parent else 0.0)) for row in instance.candidates]
    result = maximum_branching(instance.nodes, candidates, roots)
    return result.parents, sum(lookup[(child, parent)] for child, parent in result.parents.items())


def _measure(instance: SyntheticInstance) -> dict:
    initial, initial_score = _independent(instance)
    outputs = {"independent": (initial, initial_score, 0.0)}
    for name, runner in (("greedy", _greedy), ("minimal_edit", _minimal_edit)):
        started = time.perf_counter()
        choices, score = runner(instance, initial)
        outputs[name] = (choices, score, time.perf_counter() - started)
    started = time.perf_counter()
    exact = maximum_branching(instance.nodes, instance.candidates, instance.roots)
    outputs["exact"] = (exact.parents, exact.score, time.perf_counter() - started)
    exact_score = exact.score
    rows = {}
    for name, (choices, score, runtime) in outputs.items():
        graph = _graph(instance.nodes, choices)
        valid = bool(nx.is_directed_acyclic_graph(graph))
        rows[name] = {
            "valid": valid,
            "valid_rate": float(valid),
            "score": float(score),
            "objective_gap_to_exact_signed": float(exact_score - score),
            "additive_regret": max(0.0, float(exact_score - score)) if valid else None,
            "choice_edits": sum(initial[node] != choices[node] for node in instance.nodes),
            "semantic_edge_error": statistics.fmean(choices[node] != instance.gold[node] for node in instance.nodes),
            "injected_semantic_error": statistics.fmean(choices[node] != instance.gold[node] for node in instance.injected_nodes),
            "runtime_seconds": runtime,
        }
    return {"initial_cycle_count": len(_cycle_components(_graph(instance.nodes, initial))), "systems": rows}


def _summary(values: list[float | None]) -> dict:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"mean": None, "sd": None, "min": None, "max": None, "defined_n": 0}
    return {"mean": statistics.fmean(finite), "sd": statistics.stdev(finite) if len(finite) > 1 else 0.0,
            "min": min(finite), "max": max(finite), "defined_n": len(finite)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raw = args.protocol.read_bytes()
    cfg = json.loads(raw)
    sizes = [min(cfg["graph_sizes"])] if args.smoke else cfg["graph_sizes"]
    lengths = [min(cfg["cycle_lengths"])] if args.smoke else cfg["cycle_lengths"]
    pressures = [max(cfg["pressure_fractions"])] if args.smoke else cfg["pressure_fractions"]
    modes = cfg["conflict_modes"][:1] if args.smoke else cfg["conflict_modes"]
    replications = 2 if args.smoke else int(cfg["replications_per_cell"])
    cells, raw_rows = {}, []
    base_seed = int(cfg["random_seed"])
    for size in sizes:
        for length in lengths:
            for pressure in pressures:
                for mode in modes:
                    key = f"n{size}_l{length}_p{pressure:g}_{mode}"
                    rows = []
                    for replication in range(replications):
                        stable_seed = int.from_bytes(hashlib.sha256(f"{base_seed}:{key}:{replication}".encode()).digest()[:8], "big")
                        instance = generate_instance(size, length, pressure, mode, stable_seed, int(cfg["candidate_distractors_per_child"]))
                        measured = _measure(instance)
                        rows.append(measured)
                        raw_rows.append({"cell": key, "replication": replication, **measured})
                    cells[key] = {name: {metric: _summary([row["systems"][name][metric] for row in rows])
                                                 for metric in cfg["metrics"]}
                                  for name in cfg["systems"]}
                    cells[key]["initial_cycle_count"] = _summary([float(row["initial_cycle_count"]) for row in rows])
    aggregate = {name: {metric: _summary([row["systems"][name][metric] for row in raw_rows])
                        for metric in cfg["metrics"]} for name in cfg["systems"]}
    result = {"schema_version": "taxobranch.constraint_pressure_results.1", "protocol": str(args.protocol),
              "protocol_sha256": hashlib.sha256(raw).hexdigest(), "smoke": args.smoke, "cells": cells,
              "aggregate": aggregate, "raw": raw_rows,
              "audit": {"real_benchmark_test_labels_used": False, "transformer_forwards": 0,
                        "all_exact_valid": all(row["systems"]["exact"]["valid"] for row in raw_rows),
                        "all_exact_regret_zero": all(row["systems"]["exact"]["additive_regret"] < 1e-8 for row in raw_rows)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "cells": len(cells), "trials": len(raw_rows), "smoke": args.smoke}))


if __name__ == "__main__":
    main()
