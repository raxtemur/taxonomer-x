"""
Unified TExEval-2 metrics reader for Systems #1..#5 (plan §3.1, Table 1).

Every constructor / baseline in this project now emits a taxonomy JSON of
the same shape:

    {
        "system": "<name>",
        "domain": "<texeval_domain>",
        "solver_stats": {...},
        "graph_stats":  {...},
        "edges": [{"child": str, "parent": str, "score": float, "rank": int}, ...]
    }

This module loads such files, matches them against the TExEval-2 gold
taxonomy for the corresponding domain, and computes the two metrics
reported in the paper:

  * Edge-F1     — micro P/R/F1 over the predicted vs. gold edge set
                  {(child, parent)}.
  * Ancestor-F1 — for each node v, let A(v) be its ancestor set (transitive
                  closure of parent edges, reachable via BFS to tolerate
                  cycles in degenerate baselines). Micro-aggregate:
                      TP = Σ_v |A_pred(v) ∩ A_gold(v)|
                      FP = Σ_v |A_pred(v) \\ A_gold(v)|
                      FN = Σ_v |A_gold(v) \\ A_pred(v)|
                  This is the standard SemEval-2016 Task 13 metric.

Why BFS reachability and not DAG ancestry: several baselines (cosine top-1,
neural top-1, sometimes ILP when constraints are relaxed) produce cyclic
graphs. Using BFS reachability is the honest interpretation — a node in a
cycle is an "ancestor" of everything else in its SCC, which the baseline
implicitly claims. Gold is acyclic so gold ancestry is unambiguous.

Edge direction convention: edges point from child to parent, matching the
format written by `greedy_constructor.py`, `ilp_constructor.py`,
`run_proposer_to_texeval.py`, and `run_texeval_cosine.py`.

Usage:

    .venv/bin/python -m src.eval.texeval_metrics \\
        --paths data/texeval_runs_cosine/taxonomy_science_cosine.json \\
                data/texeval_runs/taxonomy_science_neural.json \\
        --out results/table1.json

    .venv/bin/python -m src.eval.texeval_metrics \\
        --dir data/texeval_runs --glob 'taxonomy_*_neural.json' \\
        --out results/neural_all_domains.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from src.data.texeval_loader import DATASETS, gold_edges_as_set, load_terms


EdgeSet = set[tuple[str, str]]


def _reachable_ancestors(edges: EdgeSet, nodes: Iterable[str]) -> dict[str, set[str]]:
    """BFS reachability on child->parent graph. Tolerates cycles.

    For each start node v, returns every node reachable from v by following
    child->parent edges, excluding v itself. In an acyclic gold graph this
    is exactly the ancestor set; in a cyclic predicted graph it is the
    transitive closure under the same semantics.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    node_set: set[str] = set(nodes)
    for c, p in edges:
        adj[c].append(p)
        node_set.add(c)
        node_set.add(p)

    anc: dict[str, set[str]] = {}
    for start in node_set:
        seen: set[str] = set()
        q: deque[str] = deque(adj.get(start, ()))
        while q:
            nxt = q.popleft()
            if nxt in seen or nxt == start:
                continue
            seen.add(nxt)
            for p in adj.get(nxt, ()):
                if p not in seen:
                    q.append(p)
        anc[start] = seen
    return anc


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def edge_metrics(pred: EdgeSet, gold: EdgeSet) -> dict[str, float]:
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    return _prf(tp, fp, fn)


def ancestor_metrics(
    pred_edges: EdgeSet,
    gold_edges: EdgeSet,
    universe: Iterable[str],
) -> dict[str, float]:
    """Micro-aggregated ancestor P/R/F1 over the union of nodes."""
    node_set = set(universe)
    for c, p in pred_edges:
        node_set.add(c)
        node_set.add(p)
    for c, p in gold_edges:
        node_set.add(c)
        node_set.add(p)

    anc_pred = _reachable_ancestors(pred_edges, node_set)
    anc_gold = _reachable_ancestors(gold_edges, node_set)

    tp = fp = fn = 0
    for v in node_set:
        ap = anc_pred.get(v, set())
        ag = anc_gold.get(v, set())
        tp += len(ap & ag)
        fp += len(ap - ag)
        fn += len(ag - ap)
    return _prf(tp, fp, fn)


def _load_taxonomy(path: Path) -> dict:
    payload = json.loads(path.read_text())
    edges = {
        (e["child"], e["parent"]) for e in payload.get("edges", [])
    }
    payload["_edge_set"] = edges
    return payload


_FILENAME_RE = re.compile(r"taxonomy_(?P<domain>.+)_(?P<system>[^_]+(?:_v\d+)?)\.json$")


def _infer_domain_and_system(path: Path, payload: dict) -> tuple[str, str]:
    domain = payload.get("domain")
    system = payload.get("system")
    if domain and system:
        return str(domain), str(system)
    m = _FILENAME_RE.match(path.name)
    if m:
        return m.group("domain"), m.group("system")
    raise ValueError(f"Cannot infer (domain, system) for {path}")


def evaluate_file(path: Path) -> dict:
    payload = _load_taxonomy(path)
    domain, system = _infer_domain_and_system(path, payload)
    if domain not in DATASETS:
        raise ValueError(
            f"Domain {domain!r} not in TExEval DATASETS "
            f"(file={path}). Known: {list(DATASETS.keys())}"
        )
    gold = gold_edges_as_set(domain)
    gold_terms = set(load_terms(domain))

    pred: EdgeSet = payload["_edge_set"]
    edge = edge_metrics(pred, gold)
    anc = ancestor_metrics(pred, gold, gold_terms)

    gs = payload.get("graph_stats", {}) or {}
    return {
        "path": str(path),
        "system": system,
        "domain": domain,
        "n_pred_edges": len(pred),
        "n_gold_edges": len(gold),
        "n_gold_terms": len(gold_terms),
        "is_dag": gs.get("is_dag"),
        "n_roots": gs.get("n_roots"),
        "edge": edge,
        "ancestor": anc,
    }


def _aggregate(rows: list[dict]) -> dict:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_system[r["system"]].append(r)

    summary: dict[str, dict] = {}
    for system, items in by_system.items():
        # Macro: mean over domains.
        def _macro(field: str, metric: str) -> float:
            vals = [it[field][metric] for it in items]
            return sum(vals) / len(vals) if vals else 0.0

        # Micro: sum counts, recompute P/R/F1.
        def _micro(field: str) -> dict[str, float]:
            tp = sum(it[field]["tp"] for it in items)
            fp = sum(it[field]["fp"] for it in items)
            fn = sum(it[field]["fn"] for it in items)
            return _prf(tp, fp, fn)

        summary[system] = {
            "n_domains": len(items),
            "macro": {
                "edge_f1": _macro("edge", "f1"),
                "edge_precision": _macro("edge", "precision"),
                "edge_recall": _macro("edge", "recall"),
                "ancestor_f1": _macro("ancestor", "f1"),
                "ancestor_precision": _macro("ancestor", "precision"),
                "ancestor_recall": _macro("ancestor", "recall"),
            },
            "micro": {
                "edge": _micro("edge"),
                "ancestor": _micro("ancestor"),
            },
        }
    return summary


def _fmt_row(r: dict) -> str:
    e = r["edge"]
    a = r["ancestor"]
    return (
        f"{r['system']:<14s} {r['domain']:<20s} "
        f"edges={r['n_pred_edges']:>5d}/{r['n_gold_edges']:<5d} "
        f"DAG={str(r['is_dag']):<5s} "
        f"eP={e['precision']:.3f} eR={e['recall']:.3f} eF1={e['f1']:.3f}  "
        f"aP={a['precision']:.3f} aR={a['recall']:.3f} aF1={a['f1']:.3f}"
    )


def _collect_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.paths:
        paths.extend(Path(p) for p in args.paths)
    if args.dir:
        root = Path(args.dir)
        paths.extend(sorted(root.glob(args.glob)))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    if not deduped:
        raise SystemExit("No taxonomy files provided. Use --paths or --dir/--glob.")
    return deduped


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--paths", nargs="*", default=None,
                   help="Explicit taxonomy JSON paths")
    p.add_argument("--dir", default=None,
                   help="Directory to scan for taxonomy files")
    p.add_argument("--glob", default="taxonomy_*.json",
                   help="Glob pattern inside --dir (default: taxonomy_*.json)")
    p.add_argument("--out", default=None,
                   help="Optional JSON file to write per-row + aggregate results")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = _collect_paths(args)

    rows: list[dict] = []
    for path in paths:
        try:
            row = evaluate_file(path)
        except Exception as exc:
            print(f"[error] {path}: {exc}")
            continue
        rows.append(row)
        print(_fmt_row(row))

    if not rows:
        raise SystemExit("No successful evaluations.")

    summary = _aggregate(rows)
    print("\n=== Aggregate (macro over domains per system) ===")
    for system, s in summary.items():
        m = s["macro"]
        mi_e = s["micro"]["edge"]
        mi_a = s["micro"]["ancestor"]
        print(
            f"  {system:<14s} n={s['n_domains']}  "
            f"macro eF1={m['edge_f1']:.3f} aF1={m['ancestor_f1']:.3f}  |  "
            f"micro eF1={mi_e['f1']:.3f} aF1={mi_a['f1']:.3f}"
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"rows": rows, "summary": summary}, indent=2, ensure_ascii=False)
        )
        print(f"\n[done] wrote {out_path}")


if __name__ == "__main__":
    main()
