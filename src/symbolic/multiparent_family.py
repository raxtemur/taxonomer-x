"""Nested forest/DAG family paths and auditable structural certificates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from src.symbolic.multiparent_projection import (
    CardinalityScores,
    DagCandidate,
    DagProjectionResult,
    Edge,
    PairScores,
    project_multiparent_dag,
)


@dataclass(frozen=True)
class CardinalityFrontierPoint:
    extra_parent_count: int
    unpenalized_score: float
    lower_penalty: float
    upper_penalty: float
    supported: bool
    result: DagProjectionResult

    def penalized_score(self, penalty: float) -> float:
        return self.unpenalized_score - penalty * self.extra_parent_count


@dataclass(frozen=True)
class FamilySelectionCertificate:
    selected_family: str
    selected: DagProjectionResult
    forest: DagProjectionResult
    best_other_family_score: float | None
    relaxation_value: float
    family_margin: float
    robustness_radius_linf: float
    decision_distance_bound: int
    raw_relaxation_value: float
    nestedness_tolerance: float
    numerical_correction_applied: bool


@dataclass(frozen=True)
class NecessityCertificate:
    edge_necessity: dict[Edge, float]
    selected_multi_parent_necessity: dict[str, float]
    candidate_multi_parent_inclusion: dict[str, float]


def _common_kwargs(
    *,
    cardinality_scores: CardinalityScores | None,
    pair_scores: PairScores | None,
    multi_parent_candidate_edges: Iterable[Edge] | None,
    solver_name: str,
    time_limit: float,
) -> dict:
    return {
        "cardinality_scores": cardinality_scores,
        "pair_scores": pair_scores,
        "multi_parent_candidate_edges": multi_parent_candidate_edges,
        "solver_name": solver_name,
        "time_limit": time_limit,
    }


def _interval(
    count: int, scores: Mapping[int, float]
) -> tuple[float, float, bool]:
    value = scores[count]
    lower = max(
        (
            (other_value - value) / (other - count)
            for other, other_value in scores.items()
            if other > count
        ),
        default=float("-inf"),
    )
    upper = min(
        (
            (value - other_value) / (count - other)
            for other, other_value in scores.items()
            if other < count
        ),
        default=float("inf"),
    )
    return lower, upper, lower <= upper + 1e-10


def solve_cardinality_frontier(
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
    root_scores: dict[str, float] | None,
    *,
    max_extra_parents: int,
    cardinality_scores: CardinalityScores | None = None,
    pair_scores: PairScores | None = None,
    multi_parent_candidate_edges: Iterable[Edge] | None = None,
    solver_name: str = "highs",
    time_limit: float = 300.0,
) -> tuple[CardinalityFrontierPoint, ...]:
    """Solve every feasible exact-cardinality problem and its penalty interval."""
    if max_extra_parents < 0:
        raise ValueError("max_extra_parents must be non-negative")
    node_values = tuple(nodes)
    candidate_values = tuple(candidates)
    solved: dict[int, DagProjectionResult] = {}
    for count in range(max_extra_parents + 1):
        try:
            solved[count] = project_multiparent_dag(
                node_values,
                candidate_values,
                root_scores,
                max_parents=2,
                second_parent_threshold=0.0,
                multi_parent_cap=max_extra_parents,
                exact_multi_parent_count=count,
                **_common_kwargs(
                    cardinality_scores=cardinality_scores,
                    pair_scores=pair_scores,
                    multi_parent_candidate_edges=multi_parent_candidate_edges,
                    solver_name=solver_name,
                    time_limit=time_limit,
                ),
            )
        except RuntimeError as error:
            if "Infeasible" not in str(error):
                raise
    if not solved or 0 not in solved:
        raise RuntimeError("forest point is infeasible")
    scores = {count: result.unpenalized_score for count, result in solved.items()}
    points = []
    for count, result in sorted(solved.items()):
        lower, upper, supported = _interval(count, scores)
        points.append(
            CardinalityFrontierPoint(
                extra_parent_count=count,
                unpenalized_score=result.unpenalized_score,
                lower_penalty=lower,
                upper_penalty=upper,
                supported=supported,
                result=result,
            )
        )
    return tuple(points)


def select_frontier_point(
    frontier: Iterable[CardinalityFrontierPoint], penalty: float
) -> CardinalityFrontierPoint:
    points = tuple(frontier)
    if not points:
        raise ValueError("frontier must not be empty")
    return min(
        points,
        key=lambda point: (
            -point.penalized_score(penalty),
            point.extra_parent_count,
        ),
    )


def assert_monotone_path(
    frontier: Iterable[CardinalityFrontierPoint], penalties: Iterable[float]
) -> None:
    ordered = sorted(float(value) for value in penalties)
    counts = [select_frontier_point(frontier, value).extra_parent_count for value in ordered]
    if any(left < right for left, right in zip(counts, counts[1:])):
        raise AssertionError("extra-parent cardinality increased with the penalty")


def _decision_distance_bound(
    selected: DagProjectionResult,
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
) -> int:
    by_child: dict[str, set[str]] = {node: set() for node in nodes}
    for candidate in candidates:
        if candidate.child in by_child:
            by_child[candidate.child].add(candidate.parent)
    total = 0
    for child in by_child:
        selected_atoms = (
            {("root", child)}
            if not selected.parents[child]
            else {("edge", parent) for parent in selected.parents[child]}
        )
        options = [{("root", child)}]
        options.extend({("edge", parent)} for parent in by_child[child])
        options.extend(
            {("edge", left), ("edge", right)}
            for left in by_child[child]
            for right in by_child[child]
            if left < right
        )
        total += max(len(selected_atoms.symmetric_difference(option)) for option in options)
    return total


def certify_family_selection(
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
    root_scores: dict[str, float] | None,
    *,
    second_parent_penalty: float,
    multi_parent_cap: int | None,
    cardinality_scores: CardinalityScores | None = None,
    pair_scores: PairScores | None = None,
    multi_parent_candidate_edges: Iterable[Edge] | None = None,
    solver_name: str = "highs",
    time_limit: float = 300.0,
) -> FamilySelectionCertificate:
    """Certify forest-vs-DAG choice and a conservative score stability radius."""
    node_values = tuple(nodes)
    candidate_values = tuple(candidates)
    common = _common_kwargs(
        cardinality_scores=cardinality_scores,
        pair_scores=pair_scores,
        multi_parent_candidate_edges=multi_parent_candidate_edges,
        solver_name=solver_name,
        time_limit=time_limit,
    )
    forest = project_multiparent_dag(
        node_values,
        candidate_values,
        root_scores,
        max_parents=2,
        second_parent_threshold=second_parent_penalty,
        multi_parent_cap=multi_parent_cap,
        exact_multi_parent_count=0,
        **common,
    )
    selected = project_multiparent_dag(
        node_values,
        candidate_values,
        root_scores,
        max_parents=2,
        second_parent_threshold=second_parent_penalty,
        multi_parent_cap=multi_parent_cap,
        **common,
    )
    raw_relaxation_value = selected.score - forest.score
    nestedness_tolerance = 1e-6 * max(
        1.0, abs(selected.score), abs(forest.score)
    )
    if raw_relaxation_value < -nestedness_tolerance:
        raise AssertionError(
            "nested DAG family scored materially below its forest subset: "
            f"gap={raw_relaxation_value:.12g}, "
            f"tolerance={nestedness_tolerance:.12g}, "
            f"dag={selected.score:.12g}, forest={forest.score:.12g}"
        )
    numerical_correction_applied = raw_relaxation_value < 0.0
    if numerical_correction_applied:
        # Forest is explicitly feasible in the max_parents=2 family.
        selected = forest
    relaxation_value = max(0.0, raw_relaxation_value)

    other_score: float | None
    if selected.extra_parent_count:
        selected_family = "dag"
        other_score = forest.score
    else:
        selected_family = "forest"
        try:
            other = project_multiparent_dag(
                node_values,
                candidate_values,
                root_scores,
                max_parents=2,
                second_parent_threshold=second_parent_penalty,
                multi_parent_cap=multi_parent_cap,
                min_multi_parent_count=1,
                **common,
            )
            other_score = other.score
        except RuntimeError as error:
            if "Infeasible" not in str(error):
                raise
            other_score = None
    margin = (
        float("inf")
        if other_score is None
        else max(0.0, selected.score - other_score)
    )
    distance = _decision_distance_bound(selected, node_values, candidate_values)
    radius = (
        float("inf")
        if math.isinf(margin)
        else margin / max(1, distance)
    )
    return FamilySelectionCertificate(
        selected_family=selected_family,
        selected=selected,
        forest=forest,
        best_other_family_score=other_score,
        relaxation_value=relaxation_value,
        raw_relaxation_value=raw_relaxation_value,
        nestedness_tolerance=nestedness_tolerance,
        numerical_correction_applied=numerical_correction_applied,
        family_margin=margin,
        robustness_radius_linf=radius,
        decision_distance_bound=distance,
    )


def necessity_certificates(
    nodes: Iterable[str],
    candidates: Iterable[DagCandidate],
    root_scores: dict[str, float] | None,
    *,
    selected: DagProjectionResult,
    second_parent_penalty: float,
    multi_parent_cap: int | None,
    inclusion_children: Iterable[str] = (),
    cardinality_scores: CardinalityScores | None = None,
    pair_scores: PairScores | None = None,
    multi_parent_candidate_edges: Iterable[Edge] | None = None,
    solver_name: str = "highs",
    time_limit: float = 300.0,
) -> NecessityCertificate:
    """Return objective gaps for forced edge/status alternatives."""
    node_values = tuple(nodes)
    candidate_values = tuple(candidates)
    common = _common_kwargs(
        cardinality_scores=cardinality_scores,
        pair_scores=pair_scores,
        multi_parent_candidate_edges=multi_parent_candidate_edges,
        solver_name=solver_name,
        time_limit=time_limit,
    )

    def solve_gap(**constraints) -> float:
        try:
            alternative = project_multiparent_dag(
                node_values,
                candidate_values,
                root_scores,
                max_parents=2,
                second_parent_threshold=second_parent_penalty,
                multi_parent_cap=multi_parent_cap,
                **constraints,
                **common,
            )
        except RuntimeError as error:
            if "Infeasible" not in str(error):
                raise
            return float("inf")
        gap = selected.score - alternative.score
        if gap < -1e-6:
            raise AssertionError("provided selected result is not globally optimal")
        return max(0.0, gap)

    edge_gaps = {
        edge: solve_gap(forbidden_edges=(edge,))
        for edge in selected.taxonomy_edges
    }
    selected_multi = {
        child: solve_gap(forbidden_multi_parent_children=(child,))
        for child in selected.multi_parent_children
    }
    requested = sorted(set(inclusion_children) - set(selected.multi_parent_children))
    inclusion = {
        child: solve_gap(required_multi_parent_children=(child,))
        for child in requested
    }
    return NecessityCertificate(
        edge_necessity=edge_gaps,
        selected_multi_parent_necessity=selected_multi,
        candidate_multi_parent_inclusion=inclusion,
    )
