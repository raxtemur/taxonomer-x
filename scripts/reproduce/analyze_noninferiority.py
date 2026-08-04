"""Run the prespecified paired non-inferiority analysis after all seeds finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _endpoint(values: np.ndarray, margin: float, alpha: float, rng, samples: int) -> dict:
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    lower_t = mean - float(stats.t.ppf(1.0 - alpha, n - 1)) * se
    boot = values[rng.integers(0, n, size=(samples, n))].mean(axis=1)
    lower_bootstrap = float(np.quantile(boot, alpha))
    return {
        "n": n,
        "mean_difference_points": mean,
        "sample_sd_points": sd,
        "standard_error_points": se,
        "noninferiority_margin_points": margin,
        "null_boundary_points": -margin,
        "one_sided_t_lower_bound_points": lower_t,
        "one_sided_bootstrap_lower_bound_points": lower_bootstrap,
        "t_test_pass": bool(lower_t > -margin),
        "bootstrap_sensitivity_pass": bool(lower_bootstrap > -margin),
        "raw_paired_differences_points": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    manifests = [_read(path) for path in args.inputs]
    expected = list(protocol["retraining_ids"])
    identifiers = [int(row.get("retraining_id", row.get("seed"))) for row in manifests]
    if identifiers != expected:
        raise RuntimeError(f"refuse partial/reordered analysis: expected={expected}, got={identifiers}")
    if any(row.get("smoke") for row in manifests):
        raise RuntimeError("smoke output cannot enter the analysis")
    expected_hash = hashlib.sha256(protocol_bytes).hexdigest()
    if any(row.get("protocol_sha256") != expected_hash for row in manifests):
        raise RuntimeError("manifest protocol hash mismatch")
    cfg = protocol["noninferiority"]
    alpha = float(cfg["alpha_one_sided"])
    samples = int(protocol["reporting"]["bootstrap_samples"])
    rng = np.random.default_rng(int(protocol["reporting"]["bootstrap_seed"]))
    differences = {}
    for metric in ("edge_f1", "ancestor_f1"):
        differences[metric] = 100.0 * np.asarray([
            float(row["macro4"]["voc_combined"][metric])
            - float(row["macro4"]["full_k20"][metric])
            for row in manifests
        ])
    endpoints = {
        "edge_f1": _endpoint(differences["edge_f1"], float(cfg["edge_f1_margin_points"]), alpha, rng, samples),
        "ancestor_f1": _endpoint(differences["ancestor_f1"], float(cfg["ancestor_f1_margin_points"]), alpha, rng, samples),
    }
    result = {
        "schema_version": "taxobranch.noninferiority_analysis.1",
        "protocol": str(args.protocol),
        "protocol_sha256": expected_hash,
        "retraining_ids": identifiers,
        "alpha_one_sided": alpha,
        "multiplicity": cfg["multiplicity"],
        "primary_test": cfg["primary_test"],
        "endpoints": endpoints,
        "coprimary_noninferiority_pass": all(row["t_test_pass"] for row in endpoints.values()),
        "bootstrap_sensitivity_both_pass": all(row["bootstrap_sensitivity_pass"] for row in endpoints.values()),
        "interpretation": "Non-inferiority is declared only if both prespecified t lower bounds exceed their negative margins.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(json.dumps({"out": str(args.out), "pass": result["coprimary_noninferiority_pass"]}))


if __name__ == "__main__":
    main()
