#!/usr/bin/env python3
"""Train one matched-objective retraining under the published budget contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL = ROOT / "configs" / "matched_objectives.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--objective", choices=("typed", "balanced_binary", "pairwise", "listwise"), required=True)
    parser.add_argument("--retraining-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    if args.retraining_id not in protocol["retraining_ids"] and not args.smoke:
        raise ValueError("retraining ID is outside the published protocol")
    cfg = protocol["training"]
    encoded_pairs = int(cfg["smoke_pair_encodings"] if args.smoke else cfg["pair_encodings_per_epoch"])

    if args.objective == "typed":
        module = "src.training.train_relation_scorer"
        command = [sys.executable, "-m", module, "--relation-objective", "typed"]
    else:
        module = "src.training.train_matched_relation_control"
        command = [sys.executable, "-m", module, "--objective", args.objective,
                   "--examples-per-epoch", str(encoded_pairs)]
        if args.objective == "listwise":
            command += ["--listwise-negatives", str(cfg["listwise_negatives"])]

    command += [
        "--out", str(args.out), "--seed", str(args.retraining_id),
        "--encoder", cfg["encoder"], "--epochs", "1" if args.smoke else str(cfg["epochs"]),
        "--batch-size", str(cfg["batch_items"][args.objective]),
        "--root-batch-size", str(cfg["root_batch_size"]),
        "--learning-rate", str(cfg["learning_rate"]), "--weight-decay", str(cfg["weight_decay"]),
        "--warmup-ratio", str(cfg["warmup_ratio"]), "--root-weight", str(cfg["root_weight"]),
        "--max-length", str(cfg["max_length"]), "--precision", cfg["precision"],
        "--num-workers", "0" if args.smoke else str(cfg["num_workers"]),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    observed = load_json(args.out / "run_config.json")
    expected_updates = encoded_pairs // int(cfg["pair_encodings_per_update"])
    observed_pairs = int(observed.get("pair_encodings_per_epoch", observed.get("train_examples", -1)))
    observed_updates = int(observed.get("optimizer_updates_per_epoch", observed.get("primary_steps_per_epoch", -1)))
    if observed_pairs != encoded_pairs or observed_updates != expected_updates:
        raise RuntimeError(
            f"matched budget violated: pairs={observed_pairs}/{encoded_pairs}, "
            f"updates={observed_updates}/{expected_updates}"
        )

    atomic_json(args.out / "reproduction_manifest.json", {
        "schema_version": "taxobranch.matched_training.1",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "objective": args.objective,
        "retraining_id": args.retraining_id,
        "training_config": observed,
        "audit": {"pair_encodings_exact": True, "optimizer_updates_exact": True},
    })


if __name__ == "__main__":
    main()
