from taxobranch import BranchCandidate, ROOT, decode_taxonomy, verify_feasibility


def test_public_decoder_returns_valid_forest():
    nodes = {"animal", "dog", "cat", "poodle"}
    candidates = [
        BranchCandidate("dog", "animal", 3.0),
        BranchCandidate("cat", "animal", 2.8),
        BranchCandidate("poodle", "dog", 3.2),
        BranchCandidate("poodle", "animal", 1.1),
    ]
    result = decode_taxonomy(
        nodes,
        candidates,
        root_scores={"animal": 2.0, "dog": 0.0, "cat": 0.0, "poodle": 0.0},
    )
    assert result.parents == {
        "animal": ROOT,
        "dog": "animal",
        "cat": "animal",
        "poodle": "dog",
    }
    assert result.roots == ["animal"]
    assert result.taxonomy_edges == [
        ("cat", "animal"),
        ("dog", "animal"),
        ("poodle", "dog"),
    ]


def test_public_feasibility_verifier_rejects_cycle():
    valid = verify_feasibility({"animal", "dog"}, {"animal": ROOT, "dog": "animal"})
    assert valid.valid
    assert valid.as_dict()["acyclic"]

    cyclic = verify_feasibility({"animal", "dog"}, {"animal": "dog", "dog": "animal"})
    assert not cyclic.valid
    assert not cyclic.acyclic
