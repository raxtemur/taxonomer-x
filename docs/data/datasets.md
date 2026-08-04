# TaxoBranch dataset protocol

The method is trained on WordNet 3.0 relation structure and evaluated on
taxonomies whose test vocabulary is removed from WordNet supervision before
examples are generated.

| Snapshot | Role | Redistribution note |
|---|---|---|
| WordNet 3.0 | five-way relation supervision; lexical-domain holdout | Preserve the Princeton WordNet license notice. |
| TExEval-2 1.2, six English domains | main external construction benchmark | Public SemEval research benchmark; cite Bordea et al. (2016). |
| TaxoLLaMA enrichment snapshot | secondary MAG-CS / MAG-PSY transfer | Derived benchmark bundle; do not commit the archive. Recreate from upstream. |
| Schema.org 30.0 | frozen OOD validation | Version 30.0 is pinned; never silently use `latest`. |

Raw and processed data are ignored by git. Download and verify them with:

```bash
python scripts/data/download_datasets.py \
  --cache-dir /path/to/optional/archive/cache
python scripts/data/prepare_external_benchmarks.py
python scripts/data/build_relation_dataset.py
```

The source manifest is `configs/data_sources.json`. Every raw file has an
expected SHA-256. `data/raw/DOWNLOAD_RECEIPT.json` records the files actually
used in a run.

## Leakage contract

1. Normalize whitespace, underscores, hyphens, and case before matching.
2. Remove every WordNet synset having any lemma that matches TExEval-2 or
   Schema.org vocabulary before relation examples are generated.
3. Assign each surviving WordNet lexical domain to exactly one of train, dev,
   or test. Thus the second benchmark is both node- and lexical-domain-disjoint.
4. Derive ancestor and sibling labels only inside a split. Cross-split paths
   are never traversed.
5. Keep `food_wordnet` and `science_wordnet` in the TExEval macro for
   comparability, but report the four non-WordNet domains separately as the
   honest external subset.

The build audit must report zero normalized-node overlap between training and
both external vocabularies, and zero synset overlap among WordNet splits.
