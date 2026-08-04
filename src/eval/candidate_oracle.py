"""Candidate-retrieval oracle coverage for end-to-end taxonomy runs."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from src.data.texeval_loader import gold_edges_as_set


def candidate_oracle(payload: dict, gold: set[tuple[str, str]], score: str = "retrieval_score", ks=(1, 5, 10, 20, 50)) -> dict:
    """Measure direct-edge coverage over *all* gold edges.

    Gold terms absent from the benchmark `.terms` file remain in the
    denominator and are therefore honest misses rather than silently dropped.
    """
    proposals = {row["child"]: row.get("candidates", []) for row in payload.get("proposals", [])}
    gold_children = {child for child, _ in gold}; rows = {}
    for k in ks:
        covered = set(); covered_children = set()
        for child, candidates in proposals.items():
            parents = {item["parent"] for item in sorted(candidates, key=lambda item: float(item.get(score, float("-inf"))), reverse=True)[:k]}
            hits = {(c, p) for c, p in gold if c == child and p in parents}; covered.update(hits)
            if hits: covered_children.add(child)
        rows[str(k)] = {"edge_recall": len(covered)/max(1, len(gold)), "child_recall": len(covered_children)/max(1, len(gold_children)),
                        "covered_edges": len(covered), "eligible_edges": len(gold), "covered_children": len(covered_children),
                        "eligible_children": len(gold_children), "gold_children_missing_from_proposals": len(gold_children-set(proposals))}
    return {"dataset": payload.get("dataset") or payload.get("domain"), "score": score, "denominator": "all_gold_edges", "at_k": rows}


def evaluate_path(path: Path, score: str, ks: tuple[int, ...]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8")); domain = payload.get("dataset") or payload.get("domain")
    if not domain: raise ValueError(f"Missing dataset/domain in {path}")
    result = candidate_oracle(payload, gold_edges_as_set(domain), score, ks); result["path"] = str(path); return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--paths", type=Path, nargs="+", required=True); p.add_argument("--score", default="retrieval_score")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20, 50]); p.add_argument("--out", type=Path, required=True); args = p.parse_args()
    rows = [evaluate_path(path, args.score, tuple(args.ks)) for path in args.paths]
    macro = {str(k): {metric: sum(row["at_k"][str(k)][metric] for row in rows)/len(rows) for metric in ("edge_recall", "child_recall")} for k in args.ks}
    payload = {"runs": rows, "macro": macro}; args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
