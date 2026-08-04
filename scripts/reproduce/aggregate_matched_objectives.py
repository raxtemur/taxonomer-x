"""Aggregate all published matched-objective retrainings regardless of direction."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = ROOT / "experiments"
METRICS = {
    "texeval": ("edge_f1", "ancestor_f1"),
    "schemaorg": ("edge_f1", "ancestor_f1"),
    "native": ("MRR", "Acc@1"),
}
NATIVE_METRICS = {
    "mag_cs": ("MRR", "Acc@1"),
    "mag_psy": ("MRR", "Acc@1"),
    "noun": ("MRR", "Acc@1", "WuP", "WuP_max_over_senses"),
    "verb": ("MRR", "Acc@1", "WuP", "WuP_max_over_senses"),
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _paired(values: list[float], draws: int, seed: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = array[rng.integers(0, len(array), size=(draws, len(array)))].mean(axis=1)
    return {
        "n": len(values), "mean": float(array.mean()),
        "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "draws": draws, "unit": "paired_retraining",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/matched_objectives.json")
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/matched_objectives.json")
    parser.add_argument("--draws", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20270806)
    args = parser.parse_args(); cfg = _load(args.protocol)
    objectives, seeds = cfg["objectives"], cfg["retraining_ids"]
    values = {
        dataset: {objective: {metric: [] for metric in metrics} for objective in objectives}
        for dataset, metrics in METRICS.items()
    }
    values["native"] = {
        objective: {domain: {metric: [] for metric in NATIVE_METRICS[domain]}
                    for domain in cfg["transfer_domains"]}
        for objective in objectives
    }
    contracts = []
    for objective in objectives:
        for seed in seeds:
            manifest = _load(args.experiment_root / f"matched_objectives/{objective}/retraining_{seed}/manifest.json")
            contracts.append((objective, seed, int(manifest["training_config"].get(
                "optimizer_updates_per_epoch", manifest["training_config"].get("primary_steps_per_epoch", -1)))))
            tex = manifest["texeval"]["macro4"]["full_k20"]
            schema = manifest["schema"]["metrics"]["systems"][objective]
            values["texeval"][objective]["edge_f1"].append(float(tex["edge_f1"]))
            values["texeval"][objective]["ancestor_f1"].append(float(tex["ancestor_f1"]))
            values["schemaorg"][objective]["edge_f1"].append(float(schema["edge"]["f1"]))
            values["schemaorg"][objective]["ancestor_f1"].append(float(schema["ancestor"]["f1"]))
            if float(tex["valid_rate"]) != 1.0 or not schema["structure"]["valid"]:
                raise RuntimeError(f"invalid decoded output: {objective} seed {seed}")
            for domain in cfg["transfer_domains"]:
                metrics = manifest["native"]["results"][domain]["metrics"]
                for metric in NATIVE_METRICS[domain]:
                    values["native"][objective][domain][metric].append(float(metrics[metric]))
    expected = int(cfg["training"]["optimizer_updates_per_epoch"])
    if any(observed != expected for _, _, observed in contracts):
        raise RuntimeError(f"common-update contract violation: {contracts}")
    summaries = {
        dataset: {
            objective: {metric: _summary(rows) for metric, rows in metrics.items()}
            for objective, metrics in systems.items()
        }
        for dataset, systems in values.items() if dataset != "native"
    }
    summaries["native"] = {
        objective: {
            domain: {metric: _summary(rows) for metric, rows in metrics.items()}
            for domain, metrics in domains.items()
        }
        for objective, domains in values["native"].items()
    }
    contrasts = {}
    comparisons = [("typed", objective) for objective in objectives if objective != "typed"]
    for dataset in ("texeval", "schemaorg"):
        contrasts[dataset] = {}
        for left, right in comparisons:
            contrasts[dataset][f"{left}_minus_{right}"] = {
                metric: _paired([a - b for a, b in zip(values[dataset][left][metric], values[dataset][right][metric])],
                                args.draws, args.bootstrap_seed + len(contrasts[dataset]) * 10 + index)
                for index, metric in enumerate(METRICS[dataset])
            }
    contrasts["native"] = {}
    for domain in cfg["transfer_domains"]:
        contrasts["native"][domain] = {}
        for left, right in comparisons:
            contrasts["native"][domain][f"{left}_minus_{right}"] = {
                metric: _paired([a - b for a, b in zip(values["native"][left][domain][metric],
                                                       values["native"][right][domain][metric])],
                                args.draws, args.bootstrap_seed + 100 + len(contrasts["native"][domain]) * 10 + index)
                for index, metric in enumerate(NATIVE_METRICS[domain])
            }
    payload = {
        "schema_version": "taxobranch.matched_objectives.aggregate.1",
        "retraining_ids": seeds, "objectives": objectives, "summaries": summaries, "contrasts": contrasts,
        "training_contract": {
            "pair_encodings_per_epoch": cfg["training"]["pair_encodings_per_epoch"],
            "pair_encodings_per_update": cfg["training"]["pair_encodings_per_update"],
            "optimizer_updates_per_epoch": expected,
        },
        "audit": {
            "all_manifests_required": True, "all_outcomes_reported": True,
            "paired_unit_is_retraining_not_domain": True, "all_decoded_outputs_valid": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"status": "completed", "out": str(args.out), "manifests": len(contracts)}))


if __name__ == "__main__":
    main()
