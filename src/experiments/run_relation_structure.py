"""Score once per seed, then decode any context ablation from the same artifact."""

from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import networkx as nx
import torch
from src.data.retrieval_cache import candidates_for, load_retrieval_cache
from src.data.texeval_loader import DATASETS, load_terms
from src.eval.root_calibration import apply_root_calibration, apply_root_margin_calibration
from src.models.candidate_graph import FEATURE_NAMES, ablate_features, candidate_graph_features
from src.models.relation_aware import CandidateGraphContextualizer, RELATION_LABELS, RelationCrossEncoder, render_role_text
from src.models.retriever import RetrievalItem, RoleConditionedRetriever
from src.symbolic.maximum_branching import BranchCandidate, maximum_branching
from src.utils.precision import PRECISIONS, autocast_context


SCORE_CACHE_VERSION = 1


def _render_input(term: str, gloss: str, role: str, input_mode: str) -> str:
    if input_mode == "full":
        return render_role_text(term, gloss, role)
    if input_mode == "labels_only":
        return render_role_text(term, "", role)
    anonymous = "[NODE_" + hashlib.sha1(term.encode("utf-8")).hexdigest()[:10] + "]"
    if input_mode == "definitions_only":
        return render_role_text("[ANON]", gloss, role)
    if input_mode == "anonymized_labels":
        return render_role_text(anonymous, gloss, role)
    raise ValueError(f"Unknown input mode: {input_mode}")



def batched_local_scores(model, tokenizer, pairs, device, batch_size, max_length, precision="auto", input_mode="full"):
    representations, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start+batch_size]; children = [_render_input(c, cd, "child", input_mode) for c, _, cd, _ in chunk]; parents = [_render_input(p, pd, "parent", input_mode) for _, p, _, pd in chunk]
            inputs = tokenizer(children, parents, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with autocast_context(device, precision): output = model(inputs)
            representations.append(output["edge_repr"].float().cpu()); logits.append(output["relation_logits"].float().cpu())
    return torch.cat(representations), torch.cat(logits)


def root_logits(model, tokenizer, items, device, batch_size, max_length, precision="auto", input_mode="full"):
    values = []
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            texts = [_render_input(item.term, item.gloss, "child", input_mode) for item in items[start:start+batch_size]]
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            with autocast_context(device, precision): value = model.root_head(model.encode(inputs)).squeeze(-1)
            values.append(value.float().cpu())
    return torch.cat(values)


def load_contextualizer(path: Path, edge_dim: int, device: torch.device):
    payload = torch.load(path, map_location="cpu", weights_only=False); config = payload.get("config", {})
    model = CandidateGraphContextualizer(edge_dim, len(FEATURE_NAMES), int(config.get("hidden_dim", 256)), float(config.get("dropout", .1)))
    model.load_state_dict(payload["state_dict"]); return model.to(device).eval()


def compute_scores(args, device, *, model=None, tokenizer=None):
    """Compute one dataset cache; callers may reuse a loaded model/tokenizer."""
    if not args.checkpoint: raise ValueError("--checkpoint is required when --score-cache is absent")
    started = time.perf_counter()
    terms = list(dict.fromkeys(load_terms(args.dataset))); definitions = json.loads(args.definitions.read_text()) if args.definitions else {}; items = [RetrievalItem(t, definitions.get(t, "")) for t in terms]
    if args.retrieval_cache:
        cache = load_retrieval_cache(args.retrieval_cache); retrieved = candidates_for(cache, args.dataset, terms, args.top_k); retrieval_source = str(args.retrieval_cache)
    else:
        retrieved = RoleConditionedRetriever(args.retriever, str(device)).retrieve(items, items, args.top_k, True); retrieval_source = args.retriever
    owns_model = model is None
    model = model or RelationCrossEncoder.from_pretrained(args.checkpoint).to(device).eval()
    if tokenizer is None:
        from transformers import AutoTokenizer
        tok_dir = args.checkpoint/"tokenizer"; tokenizer = AutoTokenizer.from_pretrained(tok_dir if tok_dir.exists() else model.cfg.encoder_name)
    pairs = [(items[c].term, items[p].term, items[c].gloss, items[p].gloss) for c, values in enumerate(retrieved) for p, _ in values]
    child = torch.tensor([c for c, values in enumerate(retrieved) for _ in values]); parent = torch.tensor([p for values in retrieved for p, _ in values])
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    edge_started = time.perf_counter()
    edge_repr, relation_logits = batched_local_scores(model, tokenizer, pairs, device, args.batch_size, args.max_length, args.precision); local = model.edge_potential(relation_logits)
    edge_seconds = time.perf_counter() - edge_started
    root_started = time.perf_counter()
    raw_roots = root_logits(model, tokenizer, items, device, args.batch_size, args.max_length, args.precision)
    root_seconds = time.perf_counter() - root_started
    calibration = json.loads(args.root_calibration.read_text()) if args.root_calibration else None; roots = apply_root_calibration(raw_roots, calibration)
    elapsed = time.perf_counter() - started
    runtime = {"wall_seconds": elapsed, "edge_seconds": edge_seconds, "root_seconds": root_seconds,
               "n_pairs": len(pairs), "n_nodes": len(terms), "batch_size": args.batch_size,
               "pairs_per_second": len(pairs)/max(edge_seconds, 1e-9), "nodes_per_second": len(terms)/max(root_seconds, 1e-9),
               "model_reused": not owns_model, "device": str(device),
               "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0}
    return {"schema_version": SCORE_CACHE_VERSION, "dataset": args.dataset, "terms": terms, "retrieved": retrieved, "child_index": child,
            "parent_index": parent, "edge_repr": edge_repr.half(), "relation_logits": relation_logits.half(), "local_scores": local,
            "root_logits_raw": raw_roots, "root_logits": roots, "root_calibration": calibration, "retrieval_source": retrieval_source,
            "checkpoint": str(args.checkpoint), "precision": args.precision, "relation_labels": list(model.relation_labels),
            "relation_objective": model.cfg.relation_objective, "runtime": runtime}


def load_scores(path: Path, dataset: str):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCORE_CACHE_VERSION or payload.get("dataset") != dataset: raise ValueError(f"Invalid score cache for {dataset}: {path}")
    return payload


def parse_args():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--checkpoint", type=Path); p.add_argument("--score-cache", type=Path); p.add_argument("--write-score-cache", type=Path)
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True); p.add_argument("--retriever", default="Qwen/Qwen3-Embedding-0.6B"); p.add_argument("--retrieval-cache", type=Path); p.add_argument("--definitions", type=Path)
    p.add_argument("--context-checkpoint", type=Path); p.add_argument("--root-calibration", type=Path); p.add_argument("--root-margin-calibration", type=Path); p.add_argument("--disable-features", nargs="*", choices=FEATURE_NAMES, default=[])
    p.add_argument("--top-k", type=int, default=50); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--precision", choices=PRECISIONS, default="auto"); p.add_argument("--device", default="auto"); p.add_argument("--out", type=Path, required=True); return p.parse_args()


def build_payload(args, scored, device):
    terms = scored["terms"]; retrieved = scored["retrieved"]; child = scored["child_index"].long(); parent = scored["parent_index"].long()
    edge_repr = scored["edge_repr"].float(); relation_logits = scored["relation_logits"].float(); local = scored["local_scores"].float(); raw_roots = scored["root_logits_raw"].float(); roots = scored["root_logits"].float()
    features = ablate_features(candidate_graph_features(child, parent, local), args.disable_features); context_delta = torch.zeros_like(local)
    if args.context_checkpoint:
        contextualizer = load_contextualizer(args.context_checkpoint, edge_repr.shape[-1], device)
        with torch.inference_mode(), autocast_context(device, args.precision): context_delta = contextualizer(edge_repr.to(device), child.to(device), parent.to(device), features.to(device), n_nodes=len(terms)).float().cpu()
    edge_scores = local+context_delta
    margin_calibration = json.loads(args.root_margin_calibration.read_text()) if getattr(args, "root_margin_calibration", None) else None
    roots = apply_root_margin_calibration(roots, child, edge_scores, margin_calibration)
    final = edge_scores-roots[child]
    candidates = [BranchCandidate(terms[int(c)], terms[int(p)], float(s)) for c, p, s in zip(child, parent, edge_scores)]
    result = maximum_branching(set(terms), candidates, {term: float(roots[i]) for i, term in enumerate(terms)}); probabilities = relation_logits.softmax(-1); rows = []; cursor = 0
    relation_labels = scored.get("relation_labels", list(RELATION_LABELS))
    for c, values in enumerate(retrieved):
        candidates_row = []
        for p, retrieval_score in values:
            candidates_row.append({"parent": terms[p], "retrieval_score": retrieval_score, "local_score": float(local[cursor]), "context_delta": float(context_delta[cursor]),
                                   "edge_score": float(edge_scores[cursor]), "root_relative_score": float(final[cursor]), "relations": {label: float(probabilities[cursor, i]) for i, label in enumerate(relation_labels)}}); cursor += 1
        rows.append({"child": terms[c], "root_logit_raw": float(raw_roots[c]), "root_logit": float(roots[c]), "candidates": candidates_row})
    return {"system": "relation_structure_branching", "domain": args.dataset, "dataset": args.dataset, "decoder": "exact_maximum_branching", "top_k": len(retrieved[0]) if retrieved else 0,
               "score_cache": str(args.score_cache or args.write_score_cache or "in_memory"), "retrieval_source": scored["retrieval_source"], "precision": args.precision,
               "relation_objective": scored.get("relation_objective", "typed"), "relation_labels": relation_labels, "runtime": scored.get("runtime", {}),
               "disabled_features": args.disable_features, "root_calibration": scored["root_calibration"], "root_margin_calibration": margin_calibration, "n_nodes": len(terms), "n_edges": len(result.taxonomy_edges), "roots": result.roots,
               "graph_stats": {"is_dag": nx.is_directed_acyclic_graph(result.graph), "n_roots": len(result.roots), "n_nodes": len(terms), "n_edges": len(result.taxonomy_edges)},
               "edges": [{"child": c, "parent": p} for c, p in result.taxonomy_edges], "proposals": rows}


def main():
    args = parse_args(); device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    scored = load_scores(args.score_cache, args.dataset) if args.score_cache else compute_scores(args, device)
    if args.write_score_cache:
        args.write_score_cache.parent.mkdir(parents=True, exist_ok=True); torch.save(scored, args.write_score_cache)
    payload = build_payload(args, scored, device)
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"wrote {args.out}: nodes={payload['n_nodes']} edges={payload['n_edges']} roots={len(payload['roots'])}")


if __name__ == "__main__": main()
