"""
TExEval-2 (SemEval 2016 Task 13) data loader.

Loads terms and gold standard taxonomies for 6 English domains used in
TAXONOMER-X Tier 2 (main external benchmark, zero-shot evaluation).
"""

import os
from pathlib import Path


_REPO = Path(__file__).resolve().parents[2]
_DATA_ROOT = Path(os.environ.get("TAXONOMERX_DATA_ROOT", _REPO / "data"))
_RAW_ROOT = _DATA_ROOT / "raw" / "texeval2" / "TExEval-2_testdata_1.2"
_LEGACY_ROOT = _DATA_ROOT / "texeval2" / "TExEval-2_testdata_1.2"
TEXEVAL_ROOT = _RAW_ROOT if _RAW_ROOT.exists() else _LEGACY_ROOT

# All 6 English datasets (see docs/plans/reactive-chasing-owl.md §1.2).
DATASETS = {
    "science": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "science_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "science_en.taxo",
    },
    "science_eurovoc": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "science_eurovoc_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "science_eurovoc_en.taxo",
    },
    "science_wordnet": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "science_wordnet_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "science_wordnet_en.taxo",
    },
    "environment_eurovoc": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "environment_eurovoc_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "environment_eurovoc_en.taxo",
    },
    "food": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "food_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "food_en.taxo",
    },
    "food_wordnet": {
        "terms": TEXEVAL_ROOT / "gs_terms" / "EN" / "food_wordnet_en.terms",
        "taxo": TEXEVAL_ROOT / "gs_taxo" / "EN" / "food_wordnet_en.taxo",
    },
}


def load_terms(dataset: str) -> list[str]:
    """Load terms list. Returns list of term strings."""
    path = DATASETS[dataset]["terms"]
    terms = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                terms.append(parts[1].strip())
            else:
                terms.append(parts[0].strip())
    return terms


def load_gold_taxonomy(dataset: str) -> list[dict]:
    """Load gold standard taxonomy edges.

    Returns list of {child, parent} dicts.
    Format: relation_id<TAB>term<TAB>hypernym
    """
    path = DATASETS[dataset]["taxo"]
    edges = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                edges.append({
                    "child": parts[1].strip(),
                    "parent": parts[2].strip(),
                })
    return edges


def gold_edges_as_set(dataset: str) -> set[tuple[str, str]]:
    """Return gold edges as set of (child, parent) tuples for fast lookup."""
    edges = load_gold_taxonomy(dataset)
    return {(e["child"], e["parent"]) for e in edges}


def all_test_terms_lower() -> set[str]:
    """Union of terms across all 6 EN domains, lowercased + stripped.

    Used by overlap_filter to exclude Task-C training pairs whose child or
    parent lexically matches any TExEval-2 test term (cross-dataset leakage
    defense).
    """
    vocab: set[str] = set()
    for ds in DATASETS:
        for t in load_terms(ds):
            vocab.add(t.strip().lower())
        for e in load_gold_taxonomy(ds):
            vocab.add(e["child"].strip().lower())
            vocab.add(e["parent"].strip().lower())
    return vocab


if __name__ == "__main__":
    total_edges = 0
    for ds in DATASETS:
        terms = load_terms(ds)
        edges = load_gold_taxonomy(ds)
        gold = gold_edges_as_set(ds)
        total_edges += len(gold)
        print(f"{ds:22s}: {len(terms):5d} terms, {len(edges):5d} edges, {len(gold):5d} unique")

        term_set = set(terms)
        missing_children = sum(1 for e in edges if e["child"] not in term_set)
        missing_parents = sum(1 for e in edges if e["parent"] not in term_set)
        if missing_children or missing_parents:
            print(f"  ! missing children: {missing_children}, missing parents: {missing_parents}")

    print(f"\nTotal unique edges across 6 EN domains: {total_edges}")
    print(f"Total test vocabulary (lower, unioned): {len(all_test_terms_lower())}")
