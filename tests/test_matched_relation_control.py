import json

from src.training.train_matched_relation_control import (
    BalancedHardDataset,
    ListwiseHardDataset,
    PairwiseHardDataset,
    build_hard_groups,
    build_hard_pairs,
)


def test_hard_pairs_are_deterministic_and_balanced(tmp_path):
    rows = [
        {"child_synset": "c", "child": "c", "parent": "p", "relation": "direct"},
        {"child_synset": "c", "child": "c", "parent": "a", "relation": "ancestor"},
        {"child_synset": "c", "child": "c", "parent": "u", "relation": "unrelated"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = build_hard_pairs(path)
    second = build_hard_pairs(path)
    assert first == second
    assert first[0][1]["relation"] == "ancestor"
    assert len(BalancedHardDataset(path)) == 2
    assert len(PairwiseHardDataset(path)) == 1


def test_listwise_groups_are_unique_and_deterministic(tmp_path):
    rows = [
        {"child_synset": "c", "child": "c", "parent": "p", "relation": "direct"},
        {"child_synset": "c", "child": "c", "parent": "a", "relation": "ancestor"},
        {"child_synset": "c", "child": "c", "parent": "r", "relation": "reverse"},
        {"child_synset": "c", "child": "c", "parent": "u", "relation": "unrelated"},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = build_hard_groups(path, negatives_per_group=2)
    assert first == build_hard_groups(path, negatives_per_group=2)
    assert len(first[0]) == 3
    assert len({row["parent"] for row in first[0]}) == 3
    assert len(ListwiseHardDataset(path, negatives_per_group=2)) == 1
