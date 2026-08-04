#!/usr/bin/env python3
"""Build the minimal, allowlisted TaxoBranch public source repository."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.release.audit_public_tree import audit_tree

DEFAULT_OUTPUT = ROOT / "release" / "taxobranch-public"
PUBLIC_FILES = """
.gitignore
CONTRIBUTING.md
LICENSE
README.md
pyproject.toml
.github/workflows/ci.yml
configs/data_sources.json
configs/constraint_pressure.json
configs/matched_objectives.json
configs/noninferiority.json
containers/Dockerfile
containers/.dockerignore
containers/requirements.lock
docs/reproducibility.md
docs/data/datasets.md
examples/toy_scores.jsonl
paper/README.md
paper/acl.sty
paper/acl_natbib.bst
paper/family_references.bib
paper/main.pdf
paper/main.tex
paper/references.bib
scripts/release/__init__.py
scripts/release/audit_public_tree.py
scripts/release/build_public_repo.py
scripts/reproduce/aggregate_matched_objectives.py
scripts/reproduce/analyze_noninferiority.py
scripts/reproduce/plan_noninferiority.py
scripts/reproduce/run_constraint_pressure.py
scripts/reproduce/train_matched_objective.py
tests/test_counterfactual_certificates.py
tests/test_constraint_pressure.py
tests/test_matched_relation_control.py
tests/test_public_api.py
scripts/data/download_datasets.py
scripts/data/prepare_external_benchmarks.py
scripts/data/build_relation_dataset.py
""".split()
PUBLIC_MODULE_ROOTS = """
taxobranch/__init__.py
src/data/retrieval_cache.py
src/data/wordnet_graph.py
src/eval/candidate_oracle.py
src/eval/conflict_conditioned.py
src/eval/taxonomy_eval.py
src/eval/error_analysis.py
src/eval/multiparent_metrics.py
src/eval/multiparent_oracles.py
src/eval/paired_bootstrap.py
src/eval/root_calibration.py
src/eval/score_calibration.py
src/eval/symbolic_repair_analysis.py
src/eval/texeval_analysis.py
src/experiments/run_cached_subgraphs.py
src/experiments/run_relation_structure.py
src/experiments/symbolic_cached.py
src/experiments/tune_symbolic_policy.py
src/models/candidate_graph.py
src/models/conditional_second_parent_retrieval.py
src/models/joint_parent_set_scorer.py
src/models/multiparent_evidence.py
src/models/relation_aware.py
src/models/retriever.py
src/models/semantic_pair_reranker.py
src/symbolic/budgeted_expansion.py
src/symbolic/certified_lazy_completion.py
src/symbolic/confidence_repair.py
src/symbolic/counterfactual_certificates.py
src/symbolic/decoding_baselines.py
src/symbolic/hierarchy_acquisition.py
src/symbolic/maximum_branching.py
src/symbolic/multiparent_family.py
src/symbolic/multiparent_projection.py
src/symbolic/ordered_multiparent_projection.py
src/symbolic/residual_acquisition.py
src/symbolic/selective_expansion.py
src/symbolic/value_of_computation.py
src/training/early_stopping.py
src/training/multiparent_evidence_training.py
src/training/relation_dataset.py
src/training/root_dataset.py
src/training/structured_loss.py
src/training/symmetric_multiparent_evidence.py
src/training/train_graph_context.py
src/training/train_matched_relation_control.py
src/training/train_relation_scorer.py
src/utils/precision.py
""".split()
PUBLIC_TREES = ("paper/figures", "paper/sections")
PACKAGE_MARKERS = ("src/__init__.py", "src/data/__init__.py", "src/models/__init__.py", "src/training/__init__.py", "src/utils/__init__.py")
SKIP_SUFFIXES = {".aux", ".bbl", ".blg", ".log", ".orig", ".out", ".pyc"}
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

def module_path(name: str) -> Path | None:
    candidate = ROOT.joinpath(*name.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate.relative_to(ROOT)
    candidate = ROOT.joinpath(*name.split("."), "__init__.py")
    return candidate.relative_to(ROOT) if candidate.is_file() else None

def import_closure(initial: list[str]) -> set[Path]:
    pending = [Path(item) for item in initial]
    selected: set[Path] = set()
    while pending:
        relative = pending.pop()
        if relative in selected:
            continue
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        selected.add(relative)
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "taxobranch" or name.startswith(("src.", "taxobranch.")):
                    dependency = module_path(name)
                    if dependency is not None and dependency not in selected:
                        pending.append(dependency)
    return selected

def copy_file(relative: str | Path, output: Path) -> None:
    relative = Path(relative)
    source = ROOT / relative
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def copy_tree(relative: str, output: Path) -> None:
    source_root = ROOT / relative
    for source in sorted(source_root.rglob("*")):
        child = source.relative_to(ROOT)
        if not source.is_file() or source.suffix in SKIP_SUFFIXES:
            continue
        if any(part in SKIP_NAMES for part in child.parts):
            continue
        copy_file(child, output)

def copy_generated_numbers(output: Path) -> None:
    source = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for name in sorted(set(re.findall(r"\\input\{(generated/[^}]+)\}", source))):
        relative = Path("paper") / name
        if not relative.suffix:
            relative = relative.with_suffix(".tex")
        copy_file(relative, output)


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None

def write_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "PUBLIC_MANIFEST.json":
            continue
        payload = path.read_bytes()
        rows.append({"path": path.relative_to(output).as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    (output / "PUBLIC_MANIFEST.json").write_text(json.dumps({"project": "TaxoBranch", "version": "0.1.0", "files": rows}, indent=2) + "\n", encoding="utf-8")

def build(output: Path) -> dict[str, object]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)
    selected = {Path(item) for item in PUBLIC_FILES + list(PACKAGE_MARKERS)}
    selected.update(import_closure(PUBLIC_MODULE_ROOTS))
    for relative in sorted(selected):
        copy_file(relative, output)
    for relative in PUBLIC_TREES:
        copy_tree(relative, output)
    copy_generated_numbers(output)
    write_manifest(output)
    report = audit_tree(output)
    if not report["ok"]:
        raise RuntimeError(json.dumps(report, indent=2, ensure_ascii=False))
    return report

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
