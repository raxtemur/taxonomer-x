import pytest

from src.symbolic.counterfactual_certificates import exact_decision_certificates
from src.symbolic.maximum_branching import ROOT, BranchCandidate


def test_exact_forced_gaps_reoptimize_globally():
    nodes = ["a", "b", "c"]
    candidates = [
        BranchCandidate("a", "b", 5.0),
        BranchCandidate("b", "a", 4.0),
        BranchCandidate("b", "c", 3.0),
        BranchCandidate("a", "c", 1.0),
    ]
    optimum, rows = exact_decision_certificates(
        nodes, candidates, {"a": 0.0, "b": 0.0, "c": 2.0}, {"a": "c", "b": "a"}
    )
    assert optimum.parents == {"a": "b", "b": "c", "c": ROOT}
    assert optimum.score == pytest.approx(10.0)
    # Excluding/forcing a->c re-optimizes b to a, giving score 7 (gap 3).
    assert rows["a"].selected_exclusion_gap == pytest.approx(3.0)
    assert rows["a"].boundary_inclusion_gap == pytest.approx(3.0)
    # Forcing b->a also reassigns a to c, giving score 7 (gap 3).
    assert rows["b"].boundary_inclusion_gap == pytest.approx(3.0)


def test_root_choice_can_be_forbidden_and_forced():
    nodes = ["a", "b"]
    candidates = [BranchCandidate("a", "b", 2.0), BranchCandidate("b", "a", 1.0)]
    optimum, rows = exact_decision_certificates(nodes, candidates, {"a": 0.0, "b": 3.0}, {"a": ROOT})
    assert optimum.parents["b"] == ROOT
    # Without the selected root for b, the best feasible structure scores 1.
    assert rows["b"].selected_exclusion_gap == pytest.approx(4.0)
    assert rows["a"].boundary_inclusion_gap == pytest.approx(2.0)
