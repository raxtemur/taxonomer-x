# Reproducing TaxoBranch

This document separates lightweight verification from full neural retraining.
Raw datasets, pretrained models, checkpoints, and dense score tensors are not
stored in Git.

## 1. Environment

Lightweight decoder and analysis environment:

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev,ilp]'
```

Full neural environment:

```bash
docker build -f containers/Dockerfile -t taxobranch:0.1.0 .
```

The container pins the exact package versions used for the reported experiments. GPU training was performed with BF16 autocast and FP32 master
weights.

## 2. CPU verification

```bash
.venv/bin/python -m pytest -q \
  tests/test_public_api.py \
  tests/test_counterfactual_certificates.py \
  tests/test_constraint_pressure.py

.venv/bin/python -m scripts.reproduce.run_constraint_pressure \
  --protocol configs/constraint_pressure.json \
  --out /tmp/taxobranch-pressure-smoke.json \
  --smoke
```

The first command checks the public decoder, exact forced-inclusion/exclusion
diagnostics, and the competing-cycle protocol. The neural matched-objective
contract test is run inside the pinned container after PyTorch is installed. The second exercises exact,
greedy, and minimal-edit decoding on deliberately competing cycles.

## 3. Data preparation

Dataset sources and expected checksums are declared in
`configs/data_sources.json`.

```bash
.venv/bin/python scripts/data/download_datasets.py
.venv/bin/python scripts/data/prepare_external_benchmarks.py
.venv/bin/python scripts/data/build_relation_dataset.py
```

The build removes normalized external vocabulary from WordNet supervision
before relation examples are generated and records a machine-readable overlap
audit. See `docs/data/datasets.md` for licenses and fixed versions.

## 4. One matched training run

After data preparation, run one objective/retraining cell:

```bash
.venv/bin/python -m scripts.reproduce.train_matched_objective \
  --protocol configs/matched_objectives.json \
  --objective typed \
  --retraining-id 117 \
  --out experiments/matched_objectives/typed/retraining_117
```

Valid objectives are `typed`, `balanced_binary`, `pairwise`, and `listwise`.
The protocol fixes pair encodings per update, optimizer updates, scheduler,
candidate pools, learned ROOT model, and exact decoder.

The training command writes reproduction_manifest.json, which records the
protocol hash and verifies the matched pair and optimizer-update budget. Score
the checkpoint on the fixed benchmark inputs with the modules under
src/experiments, and combine the resulting metrics with the training
configuration in manifest.json. This separation makes evaluation resumable
and prevents a completed checkpoint from being retrained.

After all per-retraining evaluation manifests are present, aggregate them with:

```bash
.venv/bin/python -m scripts.reproduce.aggregate_matched_objectives \
  --protocol configs/matched_objectives.json \
  --experiment-root experiments --out results/matched_objectives.json
```

## 5. Non-inferiority analysis

The 30-retraining analysis consumes completed per-retraining manifests and uses the
prespecified 0.5 edge-F1 and 1.0 ancestor-F1 margins:

```bash
.venv/bin/python -m scripts.reproduce.analyze_noninferiority \
  --protocol configs/noninferiority.json \
  --inputs experiments/noninferiority/retraining_*/manifest.json \
  --out results/noninferiority.json
```

The wildcard above is shell expansion over retraining IDs 87–116. The analyzer checks
the exact pair budget, both one-sided lower bounds, validity, and resource
fields before writing the decision.

## 6. Paper

Paper numbers are generated from validated JSON manifests under
`paper/generated/`; they are not manually transcribed. Build the PDF from
`paper/` with Tectonic:

```bash
cd paper
tectonic main.tex --keep-logs
```

The numbered main paper ends on page 8; references and supplementary material
follow.

## 7. Artifact integrity

Audit and build a content-addressed source release:

```bash
.venv/bin/python scripts/release/audit_public_tree.py .
.venv/bin/python scripts/release/build_public_repo.py --output ../taxobranch-release
```

A local checksum proves file integrity, not preregistration. Record the DOI and
immutable timestamp after uploading the final bundle to Zenodo or OSF.
