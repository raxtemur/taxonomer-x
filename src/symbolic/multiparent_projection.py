"""Exact projection onto an acyclic bounded-parent taxonomy family.

The solver consumes additive utilities. A variable-cardinality caller should
use calibrated edge log odds or an explicitly trained set model, not raw
log-probability values. Optional cardinality and parent-pair potentials make
the objective suitable for zero/one/two-parent prediction.

The returned regret is projection tension: the score paid to enforce
acyclicity and a global budget relative to independent per-child decoding
under the identical objective. It is not the value of relaxing a forest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping

import networkx as nx
import numpy as np
import pulp


ROOT = "__ROOT__"
Edge = tuple[str, str]
PairKey = tuple[str, str, str]
CardinalityScores = Mapping[str, Mapping[int, float]]
PairScores = Mapping[PairKey, float]

# Family certificates compare independently solved nested MILPs. HiGHS default
# relative MIP gap (1e-4) is too loose for that comparison: a nominally optimal
# relaxed DAG incumbent can score below an already solved forest incumbent. Keep
# the solver gap comfortably below the certificate relative tolerance (1e-6).
CERTIFIED_MIP_REL_GAP = 1e-8
_LAST_FOREST_WARM_START = None


class _WarmStartHiGHS(pulp.HiGHS):
    def __init__(self, *, warm_start: Mapping[str, float], **kwargs):
        super().__init__(**kwargs)
        self.warm_start = dict(warm_start)

    def callSolver(self, lp):
        if self.warm_start:
            variables = [
                variable
                for variable in lp.variables()
                if variable.name in self.warm_start
            ]
            indices = np.asarray(
                [variable.index for variable in variables], dtype=np.int32
            )
            values = np.asarray(
                [self.warm_start[variable.name] for variable in variables],
                dtype=np.float64,
            )
            lp.solverModel.setSolution(len(indices), indices, values)
        super().callSolver(lp)


@dataclass(frozen=True)
class DagCandidate:
    child: str
    parent: str
    score: float


@dataclass(frozen=True)
class DagProjectionResult:
    parents: dict[str, tuple[str, ...]]
    local_parents: dict[str, tuple[str, ...]]
    score: float
    local_score: float
    regret: float
    graph: nx.DiGraph
    changed_children: tuple[str, ...]
    score_sacrifice: dict[str, float]
    roots: tuple[str, ...]
    solver_status: str
    solver_name: str
    solve_seconds: float
    max_parents: int
    second_parent_threshold: float
    multi_parent_cap: int | None
    unpenalized_score: float
    extra_parent_count: int

    @property
    def taxonomy_edges(self) -> list[Edge]:
        return sorted(
            (child, parent)
            for child, parents in self.parents.items()
            for parent in parents
        )

    @property
    def multi_parent_children(self) -> tuple[str, ...]:
        return tuple(
            sorted(child for child, parents in self.parents.items() if len(parents) > 1)
        )


def _solver(
    name: str,
    time_limit: float,
    verbose: bool,
    warm_start: Mapping[str, float] | None = None,
):
    normalized = name.lower()
    if normalized == "highs":
        return _WarmStartHiGHS(
            warm_start=warm_start or {},
            msg=verbose,
            timeLimit=time_limit,
            gapRel=CERTIFIED_MIP_REL_GAP,
        )
    if normalized == "cbc":
        return pulp.PULP_CBC_CMD(
            msg=verbose, timeLimit=time_limit, gapRel=CERTIFIED_MIP_REL_GAP
        )
    raise ValueError("solver_name must be 'highs' or 'cbc'")


def _deduplicate_candidates(
    vocabulary: set[str], candidates: Iterable[DagCandidate]
) -> list[DagCandidate]:
    best: dict[Edge, float] = {}
    for candidate in candidates:
        if candidate.child == candidate.parent:
            continue
        if candidate.child not in vocabulary or candidate.parent not in vocabulary:
            continue
        key = (candidate.child, candidate.parent)
        best[key] = max(float(candidate.score), best.get(key, float("-inf")))
    return [
        DagCandidate(child, parent, score)
        for (child, parent), score in sorted(best.items())
    ]


def _cardinality(scores: CardinalityScores, child: str, count: int) -> float:
    return float(scores.get(child, {}).get(count, 0.0))


def _pair_key(child: str, left: str, right: str) -> PairKey:
    left, right = sorted((left, right))
    return child, left, right


def _pair_score(scores: PairScores, child: str, left: str, right: str) -> float:
    return float(scores.get(_pair_key(child, left, right), 0.0))


def _choice_value(
    child: str,
    parents: tuple[str, ...],
    edge_scores: Mapping[Edge, float],
    root_scores: Mapping[str, float],
    cardinality_scores: CardinalityScores,
    pair_scores: PairScores,
    second_parent_penalty: float,
) -> float:
    if not parents:
        value = float(root_scores.get(child, 0.0))
    else:
        value = sum(edge_scores[(child, parent)] for parent in parents)
    value += _cardinality(cardinality_scores, child, len(parents))
    if len(parents) == 2:
        value += _pair_score(pair_scores, child, *parents)
        value -= second_parent_penalty
    return value


def _independent_choices(
    nodes: list[str],
    by_child: Mapping[str, list[DagCandidate]],
    root_scores: Mapping[str, float],
    max_parents: int,
    second_parent_penalty: float,
    cardinality_scores: CardinalityScores,
    pair_scores: PairScores,
) -> tuple[dict[str, tuple[str, ...]], dict[str, float]]:
    choices: dict[str, tuple[str, ...]] = {}
    values: dict[str, float] = {}
    edge_scores = {
        (candidate.child, candidate.parent): candidate.score
        for rows in by_child.values()
        for candidate in rows
    }
    for child in nodes:
        rows = sorted(by_child.get(child, ()), key=lambda row: row.parent)
        options: list[tuple[str, ...]] = [()]
        options.extend((row.parent,) for row in rows)
        if max_parents > 1:
            options.extend(
                tuple(sorted((left.parent, right.parent)))
                for left, right in combinations(rows, 2)
            )
        scored = [
            (
                parents,
                _choice_value(
                    child,
                    parents,
                    edge_scores,
                    root_scores,
                    cardinality_scores,
                    pair_scores,
                    second_parent_penalty,
                ),
            )
            for parents in options
        ]
        parents, value = min(scored, key=lambda item: (-item[1], len(item[0]), item[0]))
        choices[child] = parents
        values[child] = value
    return choices, values


def project_multiparent_dag(
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
    root_scores: dict[str, float] | None = None,
    *,
    max_parents: int = 2,
    second_parent_threshold: float = 0.0,
    multi_parent_cap: int | None = None,
    cardinality_scores: CardinalityScores | None = None,
    pair_scores: PairScores | None = None,
    exact_multi_parent_count: int | None = None,
    min_multi_parent_count: int | None = None,
    required_edges: Iterable[Edge] = (),
    forbidden_edges: Iterable[Edge] = (),
    multi_parent_candidate_edges: Iterable[Edge] | None = None,
    required_multi_parent_children: Iterable[str] = (),
    forbidden_multi_parent_children: Iterable[str] = (),
    solver_name: str = "highs",
    time_limit: float = 300.0,
    verbose: bool = False,
) -> DagProjectionResult:
    """Return an exact minimum-tension projection onto a bounded-parent DAG."""
    global _LAST_FOREST_WARM_START
    vocabulary = set(nodes)
    if not vocabulary:
        raise ValueError("nodes must not be empty")
    if max_parents not in (1, 2):
        raise ValueError("this extension supports max_parents=1 or max_parents=2")
    if multi_parent_cap is not None and multi_parent_cap < 0:
        raise ValueError("multi_parent_cap must be non-negative")
    if exact_multi_parent_count is not None and exact_multi_parent_count < 0:
        raise ValueError("exact_multi_parent_count must be non-negative")
    if min_multi_parent_count is not None and min_multi_parent_count < 0:
        raise ValueError("min_multi_parent_count must be non-negative")
    if exact_multi_parent_count is not None and min_multi_parent_count is not None:
        raise ValueError("exact and minimum multi-parent counts are mutually exclusive")

    ordered_nodes = sorted(vocabulary)
    roots = {node: float((root_scores or {}).get(node, 0.0)) for node in ordered_nodes}
    cardinalities = cardinality_scores or {}
    pairs = {
        _pair_key(child, left, right): float(value)
        for (child, left, right), value in (pair_scores or {}).items()
    }
    edges = _deduplicate_candidates(vocabulary, candidates)
    family_key = (
        tuple(ordered_nodes),
        tuple((edge.child, edge.parent, edge.score) for edge in edges),
        tuple(sorted(roots.items())),
        tuple(
            (
                child,
                tuple(
                    sorted(
                        (int(count), float(value))
                        for count, value in scores.items()
                    )
                ),
            )
            for child, scores in sorted(cardinalities.items())
        ),
        tuple(sorted(pairs.items())),
        float(second_parent_threshold),
        multi_parent_cap,
        tuple(sorted(multi_parent_candidate_edges or ())),
        max_parents,
    )
    unconstrained_family = (
        not required_edges
        and not forbidden_edges
        and not required_multi_parent_children
        and not forbidden_multi_parent_children
    )
    warm_info = (
        _LAST_FOREST_WARM_START
        if exact_multi_parent_count is None
        and min_multi_parent_count is None
        and unconstrained_family
        and _LAST_FOREST_WARM_START is not None
        and _LAST_FOREST_WARM_START[0] == family_key
        else None
    )
    by_child: dict[str, list[DagCandidate]] = {node: [] for node in ordered_nodes}
    for edge in edges:
        by_child[edge.child].append(edge)
    edge_scores = {(edge.child, edge.parent): edge.score for edge in edges}

    local_parents, local_values = _independent_choices(
        ordered_nodes,
        by_child,
        roots,
        max_parents,
        second_parent_threshold,
        cardinalities,
        pairs,
    )
    local_score = sum(local_values.values())

    problem = pulp.LpProblem("taxobranch_multiparent_projection", pulp.LpMaximize)
    edge_vars = {
        index: pulp.LpVariable(f"edge_{index}", cat="Binary")
        for index in range(len(edges))
    }
    root_vars = {
        node: pulp.LpVariable(f"root_{index}", cat="Binary")
        for index, node in enumerate(ordered_nodes)
    }
    order_vars = {
        node: pulp.LpVariable(
            f"order_{index}", lowBound=0, upBound=len(ordered_nodes) - 1
        )
        for index, node in enumerate(ordered_nodes)
    }
    extra_vars = (
        {
            node: pulp.LpVariable(f"multiparent_{index}", cat="Binary")
            for index, node in enumerate(ordered_nodes)
        }
        if max_parents > 1
        else {}
    )

    edge_indices: dict[str, list[int]] = {node: [] for node in ordered_nodes}
    edge_lookup: dict[Edge, int] = {}
    for index, edge in enumerate(edges):
        edge_indices[edge.child].append(index)
        edge_lookup[(edge.child, edge.parent)] = index

    pair_vars: dict[PairKey, pulp.LpVariable] = {}
    for pair_index, key in enumerate(sorted(pairs)):
        child, left, right = key
        if (child, left) not in edge_lookup or (child, right) not in edge_lookup:
            continue
        variable = pulp.LpVariable(f"pair_{pair_index}", cat="Binary")
        pair_vars[key] = variable
        left_var = edge_vars[edge_lookup[(child, left)]]
        right_var = edge_vars[edge_lookup[(child, right)]]
        problem += variable <= left_var
        problem += variable <= right_var
        problem += variable >= left_var + right_var - 1

    objective = pulp.lpSum(
        edges[index].score * variable for index, variable in edge_vars.items()
    )
    objective += pulp.lpSum(roots[node] * root_vars[node] for node in ordered_nodes)
    if extra_vars:
        objective -= second_parent_threshold * pulp.lpSum(extra_vars.values())
    objective += pulp.lpSum(
        _cardinality(cardinalities, node, 0) * root_vars[node]
        + _cardinality(cardinalities, node, 1)
        * (1 - root_vars[node] - extra_vars.get(node, 0))
        + _cardinality(cardinalities, node, 2) * extra_vars.get(node, 0)
        for node in ordered_nodes
    )
    objective += pulp.lpSum(
        _pair_score(pairs, *key) * variable for key, variable in pair_vars.items()
    )
    problem += objective

    for child_index, child in enumerate(ordered_nodes):
        count = pulp.lpSum(edge_vars[index] for index in edge_indices[child])
        problem += count >= 1 - root_vars[child], f"nonroot_has_parent_{child_index}"
        problem += (
            count <= max_parents * (1 - root_vars[child]),
            f"root_xor_parents_{child_index}",
        )
        if extra_vars:
            problem += count - 1 <= extra_vars[child], f"extra_upper_{child_index}"
            problem += count >= 2 * extra_vars[child], f"extra_lower_{child_index}"

    if multi_parent_cap is not None and extra_vars:
        cap = min(int(multi_parent_cap), len(ordered_nodes))
        problem += pulp.lpSum(extra_vars.values()) <= cap, "multi_parent_cap"
    else:
        cap = None if multi_parent_cap is None else 0

    if exact_multi_parent_count is not None:
        if not extra_vars and exact_multi_parent_count:
            raise ValueError("positive exact multi-parent count requires max_parents=2")
        problem += pulp.lpSum(extra_vars.values()) == int(exact_multi_parent_count)
    if min_multi_parent_count is not None:
        if not extra_vars and min_multi_parent_count:
            raise ValueError("positive minimum multi-parent count requires max_parents=2")
        problem += pulp.lpSum(extra_vars.values()) >= int(min_multi_parent_count)

    required_edge_set = set(required_edges)
    forbidden_edge_set = set(forbidden_edges)
    if required_edge_set & forbidden_edge_set:
        raise ValueError("an edge cannot be both required and forbidden")
    missing_required = required_edge_set - set(edge_lookup)
    if missing_required:
        raise ValueError(
            f"required edges are absent from candidates: {sorted(missing_required)!r}"
        )
    for edge in required_edge_set:
        problem += edge_vars[edge_lookup[edge]] == 1
    for edge in forbidden_edge_set & set(edge_lookup):
        problem += edge_vars[edge_lookup[edge]] == 0

    if multi_parent_candidate_edges is not None and extra_vars:
        allowed_multi_edges = set(multi_parent_candidate_edges)
        unknown_allowed = allowed_multi_edges - set(edge_lookup)
        if unknown_allowed:
            raise ValueError(
                f"multi-parent candidate edges are absent: {sorted(unknown_allowed)!r}"
            )
        for edge, index in edge_lookup.items():
            child, _ = edge
            if edge not in allowed_multi_edges:
                problem += edge_vars[index] + extra_vars[child] <= 1

    required_multi = set(required_multi_parent_children)
    forbidden_multi = set(forbidden_multi_parent_children)
    if required_multi & forbidden_multi:
        raise ValueError("a child cannot be both required and forbidden multi-parent")
    unknown_multi = (required_multi | forbidden_multi) - vocabulary
    if unknown_multi:
        raise ValueError(f"unknown constrained children: {sorted(unknown_multi)!r}")
    if required_multi and not extra_vars:
        raise ValueError("required multi-parent children need max_parents=2")
    for child in required_multi:
        problem += extra_vars[child] == 1
    for child in forbidden_multi:
        if extra_vars:
            problem += extra_vars[child] == 0

    big_m = len(ordered_nodes)
    node_index = {node: index for index, node in enumerate(ordered_nodes)}
    for index, edge in enumerate(edges):
        problem += (
            order_vars[edge.parent]
            - order_vars[edge.child]
            + big_m * (1 - edge_vars[index])
            >= 1,
            f"acyclic_{node_index[edge.child]}_{node_index[edge.parent]}",
        )

    warm_start = {}
    if warm_info is not None:
        _, warm_parents, warm_score = warm_info
        warm_edges = {
            (child, parent)
            for child, parent_values in warm_parents.items()
            for parent in parent_values
        }
        warm_graph = nx.DiGraph()
        warm_graph.add_nodes_from(ordered_nodes)
        warm_graph.add_edges_from(warm_edges)
        warm_order = {
            node: index
            for index, node in enumerate(nx.topological_sort(warm_graph))
        }
        warm_start.update({
            variable.name: float(
                (edges[index].child, edges[index].parent) in warm_edges
            )
            for index, variable in edge_vars.items()
        })
        warm_start.update({
            variable.name: float(not warm_parents[node])
            for node, variable in root_vars.items()
        })
        warm_start.update({
            variable.name: float(len(warm_parents[node]) > 1)
            for node, variable in extra_vars.items()
        })
        warm_start.update({
            variable.name: float(
                left in warm_parents[child] and right in warm_parents[child]
            )
            for (child, left, right), variable in pair_vars.items()
        })
        warm_start.update({
            variable.name: float(warm_order[node])
            for node, variable in order_vars.items()
        })
        objective_floor_tolerance = 1e-8 * max(1.0, abs(warm_score))
        problem += (
            objective >= warm_score - objective_floor_tolerance,
            "known_forest_objective_floor",
        )

    started = time.perf_counter()
    status_code = problem.solve(
        _solver(solver_name, time_limit, verbose, warm_start)
    )
    solve_seconds = time.perf_counter() - started
    status = pulp.LpStatus.get(status_code, f"status_{status_code}")
    if status != "Optimal":
        raise RuntimeError(
            f"Multi-parent projection is not certified: solver status={status!r}"
        )

    selected_by_child: dict[str, list[str]] = {node: [] for node in ordered_nodes}
    for index, edge in enumerate(edges):
        if float(edge_vars[index].value() or 0.0) > 0.5:
            selected_by_child[edge.child].append(edge.parent)
    selected_roots = [
        node
        for node in ordered_nodes
        if float(root_vars[node].value() or 0.0) > 0.5
    ]
    parents = {
        child: tuple(sorted(parent_values))
        for child, parent_values in selected_by_child.items()
    }
    multi_children = [child for child, values in parents.items() if len(values) > 1]
    selected_values = {
        child: _choice_value(
            child,
            parents[child],
            edge_scores,
            roots,
            cardinalities,
            pairs,
            second_parent_threshold,
        )
        for child in ordered_nodes
    }
    selected_score = sum(selected_values.values())
    unpenalized_score = selected_score + second_parent_threshold * len(multi_children)

    graph = nx.DiGraph()
    graph.add_nodes_from(ordered_nodes)
    graph.add_edges_from(
        (child, parent)
        for child, parent_values in parents.items()
        for parent in parent_values
    )
    if not nx.is_directed_acyclic_graph(graph):
        raise AssertionError("Optimal ILP result violates acyclicity")
    if set(selected_roots) != {
        child for child, values in parents.items() if not values
    }:
        raise AssertionError("ROOT variables disagree with selected parent sets")

    sacrifices: dict[str, float] = {}
    changed: list[str] = []
    for child in ordered_nodes:
        sacrifice = local_values[child] - selected_values[child]
        if sacrifice < -1e-7:
            raise AssertionError(
                "Feasible child choice exceeds its independent local optimum"
            )
        sacrifices[child] = max(0.0, sacrifice)
        if parents[child] != local_parents[child]:
            changed.append(child)

    regret = local_score - selected_score
    if regret < -1e-6:
        raise AssertionError("Feasible ILP score exceeds independent optimum")
    if abs(sum(sacrifices.values()) - regret) > 1e-5 * max(1.0, abs(regret)):
        raise AssertionError("Itemized score sacrifices do not sum to projection tension")

    if (
        exact_multi_parent_count == 0
        and min_multi_parent_count is None
        and unconstrained_family
    ):
        _LAST_FOREST_WARM_START = (family_key, dict(parents), selected_score)

    return DagProjectionResult(
        parents=parents,
        local_parents=local_parents,
        score=selected_score,
        local_score=local_score,
        regret=max(0.0, regret),
        graph=graph,
        changed_children=tuple(sorted(changed)),
        score_sacrifice=sacrifices,
        roots=tuple(sorted(selected_roots)),
        solver_status=status,
        solver_name=solver_name,
        solve_seconds=solve_seconds,
        max_parents=max_parents,
        second_parent_threshold=second_parent_threshold,
        multi_parent_cap=cap,
        unpenalized_score=unpenalized_score,
        extra_parent_count=len(multi_children),
    )
