"""Paired significance and taxonomy error analysis from one decoded payload."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from src.data.texeval_loader import gold_edges_as_set, load_terms
from src.eval.error_analysis import classify_errors
from src.eval.paired_bootstrap import paired_bootstrap


def edges(payload, variant):
    row = payload["variants"][variant]; return {(edge["child"], edge["parent"]) for edge in row["edges"]}


def analyze(payload, domain, baseline="relation_top1", system="exact_branching", samples=10000, seed=42):
    gold = gold_edges_as_set(domain); a = edges(payload, baseline); b = edges(payload, system); universe = set(load_terms(domain)) | {node for edge in gold for node in edge}
    local = {child: parent for child, parent in a}; final = {child: parent for child, parent in b}; changed = {child for child in universe if local.get(child) != final.get(child)}
    return {"domain": domain, "baseline": baseline, "system": system,
            "paired_bootstrap": paired_bootstrap(a, b, gold, samples, seed), "errors": classify_errors(b, gold, universe),
            "global_overrides": {"count": len(changed), "rate": len(changed)/max(1, len(universe)), "examples": sorted(changed)[:50]}}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--decoded", type=Path, required=True); p.add_argument("--domain", required=True)
    p.add_argument("--baseline", default="relation_top1"); p.add_argument("--system", default="exact_branching"); p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    result = analyze(json.loads(args.decoded.read_text()), args.domain, args.baseline, args.system, args.samples, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")


if __name__ == "__main__": main()
