"""Train graph context with edge, listwise, structured and feature ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from src.models.candidate_graph import FEATURE_NAMES, ablate_features
from src.models.relation_aware import CandidateGraphContextualizer
from src.symbolic.maximum_branching import ROOT
from src.training.early_stopping import EarlyStopping
from src.training.structured_loss import structured_hinge


def _ancestor_costs(nodes: list[str], gold_parents: dict[str, str], alpha: float) -> dict[str, float]:
    children = {node: [] for node in nodes}
    for child, parent in gold_parents.items():
        if parent != ROOT and parent in children:
            children[parent].append(child)
    memo: dict[str, int] = {}

    def descendants(node: str, visiting: set[str]) -> int:
        if node in memo:
            return memo[node]
        if node in visiting:
            return 0
        total = sum(1 + descendants(child, visiting | {node}) for child in children[node])
        memo[node] = total
        return total

    scale = max(1, len(nodes) - 1)
    return {node: 1.0 + alpha * descendants(node, set()) / scale for node in gold_parents}


def graph_loss(model, graph, device, listwise_weight, structured_weight, disabled_features=(), ancestor_alpha=0.0):
    child = graph["child_index"].to(device); parent = graph["parent_index"].to(device)
    edge_repr = graph["edge_repr"].float().to(device); local = graph["local_scores"].to(device)
    roots = graph["root_logits"].to(device)
    features = ablate_features(graph["scalar_features"].to(device), disabled_features)
    labels = graph["labels"].to(device)
    delta = model(edge_repr, child, parent, features, n_nodes=len(graph["nodes"])); scores = local + delta - roots[child]
    positive = labels.sum().clamp_min(1.0); pos_weight = (labels.numel() - positive) / positive
    edge_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)
    listwise = scores.new_zeros(()); groups = 0
    exact_labels = "gold_parents" in graph
    gold_parents = dict(graph.get("gold_parents", {}))
    edge_pairs = [(graph["nodes"][c], graph["nodes"][p]) for c, p in zip(child.tolist(), parent.tolist())]
    for c, child_name in enumerate(graph["nodes"]):
        if exact_labels and child_name not in gold_parents:
            continue
        indices = (child == c).nonzero(as_tuple=False).flatten(); positives = indices[labels[indices] > 0.5]
        choice = int(positives[0]) if positives.numel() else -1
        logits = torch.cat((scores[indices], scores.new_zeros(1)))
        target = (indices == choice).nonzero(as_tuple=False).flatten(); target_index = int(target[0]) if target.numel() else len(indices)
        listwise += F.cross_entropy(logits.unsqueeze(0), torch.tensor([target_index], device=device)); groups += 1
        if not exact_labels:
            gold_parents[child_name] = edge_pairs[choice][1] if choice >= 0 else ROOT
    listwise /= max(1, groups); structured = scores.new_zeros(())
    if structured_weight:
        costs = _ancestor_costs(graph["nodes"], gold_parents, ancestor_alpha)
        structured = structured_hinge(
            graph["nodes"], edge_pairs, scores, scores.new_zeros(len(graph["nodes"])),
            gold_parents, node_costs=costs,
        ) / max(1, len(gold_parents))
    return edge_loss + listwise_weight * listwise + structured_weight * structured, scores



@torch.inference_mode()
def evaluate(model, graphs, device, disabled_features=()):
    model.eval(); correct = total = 0
    for graph in graphs:
        child = graph["child_index"].to(device); labels = graph["labels"].to(device)
        features = ablate_features(graph["scalar_features"].to(device), disabled_features)
        delta = model(graph["edge_repr"].float().to(device), child, graph["parent_index"].to(device), features, n_nodes=len(graph["nodes"]))
        scores = graph["local_scores"].to(device) + delta - graph["root_logits"].to(device)[child]
        for c in range(len(graph["nodes"])):
            indices = (child == c).nonzero(as_tuple=False).flatten()
            if len(indices):
                pred = indices[scores[indices].argmax()]; correct += int(labels[pred] > 0.5); total += 1
    return correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", type=Path, required=True); parser.add_argument("--dev-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10); parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--listwise-weight", type=float, default=0.2); parser.add_argument("--structured-weight", type=float, default=0.1)
    parser.add_argument("--disable-features", nargs="*", choices=FEATURE_NAMES, default=[])
    parser.add_argument("--ancestor-alpha", type=float, default=0.0)
    parser.add_argument("--graph-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default="auto")
    args = parser.parse_args(); random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    train = torch.load(args.train_cache, map_location="cpu", weights_only=False)["graphs"]
    dev = torch.load(args.dev_cache, map_location="cpu", weights_only=False)["graphs"]
    if not train or not dev:
        raise ValueError("train and dev graph caches must be non-empty")
    model = CandidateGraphContextualizer(train[0]["edge_repr"].shape[-1], len(FEATURE_NAMES), args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate); history = []
    early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train); losses = []
        for start in range(0, len(train), args.graph_batch_size):
            graphs = train[start:start + args.graph_batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_losses = []
            for graph in graphs:
                loss, _ = graph_loss(
                    model, graph, device, args.listwise_weight, args.structured_weight,
                    args.disable_features, args.ancestor_alpha,
                )
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise FloatingPointError(f"Non-finite graph loss at epoch={epoch}")
                batch_losses.append(loss)
                losses.append(float(loss.detach()))
            torch.stack(batch_losses).mean().backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not bool(torch.isfinite(grad_norm.detach()).all()):
                raise FloatingPointError(f"Non-finite graph gradient at epoch={epoch}")
            optimizer.step()
        top1 = evaluate(model, dev, device, args.disable_features)
        if not math.isfinite(top1):
            raise FloatingPointError(f"Non-finite graph dev_top1 at epoch={epoch}")
        improved, should_stop = early_stopping.update(top1, epoch)
        metrics = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_top1": top1,
                   "improved": improved, "best_dev_top1": early_stopping.best,
                   "best_epoch": early_stopping.best_epoch,
                   "epochs_without_improvement": early_stopping.bad_epochs}
        history.append(metrics); print(json.dumps(metrics), flush=True)
        if improved:
            torch.save({"config": {"hidden_dim": args.hidden_dim, "dropout": 0.1,
                                     "disabled_features": args.disable_features,
                                     "listwise_weight": args.listwise_weight,
                                     "ancestor_alpha": args.ancestor_alpha,
                                     "graph_batch_size": args.graph_batch_size,
                                     "structured_weight": args.structured_weight,
                                     "early_stopping_patience": args.early_stopping_patience,
                                     "early_stopping_min_delta": args.early_stopping_min_delta},
                        "state_dict": model.state_dict(), "dev_top1": top1,
                        "best_epoch": epoch}, args.out)
        args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
        if should_stop:
            print(json.dumps({"early_stopping": early_stopping.as_dict()}), flush=True)
            break
    args.out.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
    args.out.with_suffix(".training_summary.json").write_text(json.dumps({
        "epochs_completed": len(history), "epochs_requested": args.epochs,
        "early_stopping": early_stopping.as_dict(),
    }, indent=2) + "\n")
    if early_stopping.best_epoch == 0 or not args.out.exists():
        raise RuntimeError("Graph training completed without a finite best checkpoint")


if __name__ == "__main__":
    main()
