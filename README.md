# TaxoBranch

**Auditable taxonomy induction with typed neural evidence, exact constrained
decoding, and budgeted candidate scoring.**

TaxoBranch separates three claims that are often conflated in taxonomy
induction:

1. **Semantic evidence:** a typed DeBERTa scorer distinguishes direct parents
   from ancestors, siblings, reversed relations, and unrelated candidates.
2. **Structural feasibility:** an exact maximum-branching decoder returns the
   maximum-scoring acyclic one-parent-or-ROOT taxonomy and a machine-checkable
   feasibility certificate.
3. **Compute allocation:** a development-trained gate selectively scores
   additional candidates under an exact per-graph pair budget.

The repository accompanies the paper
[*TaxoBranch: Auditable Taxonomy Induction with Typed Evidence and Certified
Structure*](paper/main.pdf). Maximum branching and structured max-marginals
are established methods; the contribution is their auditable integration with
typed supervision, explicit failure attribution, and controlled compute
allocation.

## Main results

| Finding | Result |
|---|---:|
| Reference system on TExEval-2 Macro4 | 38.07 edge F1 / 40.87 ancestor F1 / 100% validity |
| Typed vs. matched listwise supervision | +5.80 edge F1 / +8.18 ancestor F1 |
| Typed vs. matched binary on Schema.org | +6.46 edge F1 / +8.10 ancestor F1 |
| 75% vs. full pair scoring | non-inferior within 0.5 edge-F1 and 1.0 ancestor-F1 margins |
| Measured savings at 75% pair budget | 23.69% CUDA time / 20.79% wall time |
| Competing-cycle stress suite | 100% validity and zero optimality gap over 1,280 instances |

All matched comparisons use the same backbone, candidate access, pair
encodings, optimizer updates, learned ROOT model, and exact decoder. The paper
does **not** claim backbone-independent gains or protocol-compatible
state-of-the-art performance.

## Install

TaxoBranch requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
```

The default installation contains the lightweight exact decoder. Install the
neural and ILP stacks only when needed:

```bash
.venv/bin/pip install -e '.[neural,ilp,dev]'
```

For the fully pinned GPU environment, use
`containers/Dockerfile` and
`containers/requirements.lock`.

## Five-minute example

Decode the included scored candidate graph:

```bash
taxobranch-decode \
  --scores examples/toy_scores.jsonl \
  --out /tmp/taxobranch-toy.json
```

Each JSONL row supplies a child, its ROOT potential, and scored parent
candidates:

```json
{"child":"poodle","root_score":0.0,"candidates":[{"parent":"dog","score":3.2}]}
```

Python API:

```python
from taxobranch import BranchCandidate, decode_taxonomy

result = decode_taxonomy(
    {"animal", "dog", "poodle"},
    [
        BranchCandidate("dog", "animal", 3.0),
        BranchCandidate("poodle", "dog", 3.2),
    ],
    root_scores={"animal": 2.0, "dog": 0.0, "poodle": 0.0},
)

print(result.taxonomy_edges)
# [('dog', 'animal'), ('poodle', 'dog')]
```

Public edges use the `child -> parent` convention. Edge reversal for
Chu–Liu/Edmonds is internal to the decoder.

## Verify the release

CPU-only core tests:

```bash
.venv/bin/python -m pytest -q \
  tests/test_public_api.py \
  tests/test_counterfactual_certificates.py
```

CPU-only competing-cycle smoke test:

```bash
.venv/bin/python -m scripts.reproduce.run_constraint_pressure \
  --protocol configs/constraint_pressure.json \
  --out /tmp/taxobranch-pressure.json \
  --smoke
```

Full data preparation, training, aggregation, paper regeneration, and expected
resource requirements are documented in
[`docs/reproducibility.md`](docs/reproducibility.md). Dataset versions,
checksums, and redistribution constraints are in
`docs/data/datasets.md`.

## Repository layout

```text
taxobranch/             stable public Python facade
src/
  symbolic/             exact decoder, max-marginal diagnostics, allocation
  models/               retrieval and neural models
  training/             typed and matched-objective training
  eval/                 metrics, bootstrap, ceilings, error attribution
scripts/reproduce/      public experiment entry points and aggregation
configs/                published experiment protocols
tests/                  unit and protocol tests
paper/                  LaTeX source, generated numbers, and PDF
containers/             pinned GPU environment
examples/               small runnable inputs
```

Generated checkpoints, score tensors, raw datasets, scheduler credentials, and
machine-specific job scripts are intentionally excluded from Git. Aggregated
results and protocol seals are distributed separately as a versioned research
artifact when licensing permits.

## Terminology and API stability

- A **feasibility certificate** verifies coverage, one parent-or-ROOT choice
  per node, and acyclicity.
- **Constraint tension** is the score difference between independent local
  choices and the exact feasible optimum.
- An **optimality gap** compares another feasible decoder with that optimum.
- Forced-inclusion and forced-exclusion values are exposed as structured
  max-marginal **sensitivity diagnostics**, not semantic confidence.

The `taxobranch` facade follows this terminology. Lower-level modules under `src/` implement the paper components used by the public facade.

## Data and model licenses

Code is released under the [MIT License](LICENSE). WordNet, TExEval-2,
Schema.org, MAG benchmarks, pretrained encoders, and external baselines retain
their own licenses and are not relicensed by this repository. Review the data
documentation before redistribution.

## Citation

Please cite the accompanying paper. Final BibTeX and archival DOI metadata will
be added after the anonymous review period; the public-release checklist
prevents publishing placeholder authorship as final metadata.

## Contributing

Bug reports and reproducibility issues are welcome. Please include the command,
configuration file, Python version, and the smallest failing example. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) before submitting code changes.
