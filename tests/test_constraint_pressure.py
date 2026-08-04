from scripts.reproduce.run_constraint_pressure import _measure, generate_instance


def test_competing_cycles_are_repaired_exactly():
    instance = generate_instance(
        size=32,
        length=4,
        pressure=0.5,
        mode="overlapping",
        seed=202741,
        distractors=3,
    )
    result = _measure(instance)
    assert result["initial_cycle_count"] > 0
    assert result["systems"]["independent"]["valid"] is False
    assert result["systems"]["exact"]["valid"] is True
    assert result["systems"]["exact"]["additive_regret"] == 0.0
