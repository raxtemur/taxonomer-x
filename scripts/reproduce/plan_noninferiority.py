"""Planning-only power and margin sensitivity for a future non-inferiority run.

This utility reads the already observed primary aggregate. Its output is not a
confirmatory test: margins chosen after seeing these data require a new sealed
replication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("results/allocation_precursor.json"),
    )
    parser.add_argument("--edge-margin-points", type=float, default=0.5)
    parser.add_argument("--ancestor-margin-points", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 0.5 or not 0.5 < args.power < 1.0:
        raise ValueError("invalid alpha or power")
    aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
    comparison = aggregate["paired_combined_minus"]["full_k20"]
    margins = {
        "edge_f1": args.edge_margin_points,
        "ancestor_f1": args.ancestor_margin_points,
    }
    normal = NormalDist()
    z = normal.inv_cdf(1.0 - args.alpha) + normal.inv_cdf(args.power)
    results = {}
    for metric, margin in margins.items():
        observed = comparison[metric]
        mean = 100.0 * float(observed["mean"])
        std = 100.0 * float(observed["std"])
        ci = [100.0 * float(value) for value in observed["ci95"]]
        separation = mean + margin
        required_n = None
        if separation > 0.0 and std > 0.0:
            required_n = math.ceil((z * std / separation) ** 2)
        results[metric] = {
            "observed_delta_points": mean,
            "observed_two_sided_ci95_points": ci,
            "smallest_margin_implied_by_current_two_sided_lower_bound": max(0.0, -ci[0]),
            "proposed_future_margin_points": margin,
            "normal_approx_required_new_retrainings": required_n,
        }
    print(json.dumps({
        "status": "planning_only_post_outcome_not_confirmatory",
        "assumptions": "future retraining deltas retain the observed mean and standard deviation",
        "one_sided_alpha": args.alpha,
        "target_power": args.power,
        "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
