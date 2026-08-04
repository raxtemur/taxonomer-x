"""Train compute-matched hard-negative binary, pairwise, or listwise controls.

The source pool, encoder, ROOT auxiliary task, token budget, epochs, and fixed
hard-negative construction are shared.  The saved checkpoint remains a normal
binary RelationCrossEncoder, so downstream decoding is exactly unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler

from src.models.relation_aware import RelationCrossEncoder, RelationScorerConfig, render_role_text
from src.training.early_stopping import EarlyStopping
from src.training.root_dataset import RootCollator, RootJsonlDataset
from src.training.train_relation_scorer import (
    auxiliary_batches_for_step,
    evaluate,
    evaluate_roots,
    require_finite,
    seed_everything,
)
from src.training.relation_dataset import RelationCollator, RelationJsonlDataset
from src.utils.precision import PRECISIONS, autocast_context, resolve_precision, uses_fp16


HARDNESS = {"ancestor": 4, "reverse": 3, "sibling": 2, "unrelated": 1}


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in value.replace("_", " ").split() if token}


def _hardness(row: dict) -> tuple[float, int, str]:
    child = _tokens(row.get("child", "")) | _tokens(row.get("child_gloss", ""))
    parent = _tokens(row.get("parent", "")) | _tokens(row.get("parent_gloss", ""))
    overlap = len(child & parent) / max(1, len(child | parent))
    stable = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
    return overlap, HARDNESS.get(row["relation"], 0), stable


def build_hard_pairs(path: Path) -> list[tuple[dict, dict]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_child: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_child[row["child_synset"]].append(row)
    pairs = []
    for child in sorted(by_child):
        group = by_child[child]
        positives = sorted((row for row in group if row["relation"] == "direct"), key=lambda row: json.dumps(row, sort_keys=True))
        negatives = [row for row in group if row["relation"] != "direct"]
        if not negatives:
            continue
        ordered = sorted(negatives, key=_hardness, reverse=True)
        for index, positive in enumerate(positives):
            pairs.append((positive, ordered[index % len(ordered)]))
    if not pairs:
        raise RuntimeError(f"no direct/hard-negative pairs in {path}")
    return pairs


class BalancedHardDataset(Dataset):
    def __init__(self, path: Path):
        self.pairs = build_hard_pairs(path)
        self.rows = []
        for positive, negative in self.pairs:
            self.rows.extend(({**positive, "root_label": -1.0, "target_relation": "direct", "label_id": 0},
                              {**negative, "root_label": -1.0, "target_relation": "non_direct", "label_id": 1}))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PairwiseHardDataset(Dataset):
    def __init__(self, path: Path):
        self.pairs = build_hard_pairs(path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        positive, negative = self.pairs[index]
        return {"positive": positive, "negative": negative}


def build_hard_groups(path: Path, negatives_per_group: int = 2) -> list[tuple[dict, ...]]:
    """Build deterministic direct-vs-multiple-negative groups without repeats."""
    if negatives_per_group < 2:
        raise ValueError("listwise supervision requires at least two negatives")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    by_child: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_child[row["child_synset"]].append(row)
    groups: list[tuple[dict, ...]] = []
    for child in sorted(by_child):
        rows_for_child = by_child[child]
        positives = sorted(
            (row for row in rows_for_child if row["relation"] == "direct"),
            key=lambda row: json.dumps(row, sort_keys=True),
        )
        ordered_negatives = sorted(
            (row for row in rows_for_child if row["relation"] != "direct"),
            key=_hardness,
            reverse=True,
        )
        for index, positive in enumerate(positives):
            by_parent = {}
            for negative in ordered_negatives:
                if negative["parent"] != positive["parent"]:
                    by_parent.setdefault(negative["parent"], negative)
            negatives = list(by_parent.values())
            if len(negatives) < negatives_per_group:
                continue
            selected = tuple(
                negatives[(index + offset) % len(negatives)]
                for offset in range(negatives_per_group)
            )
            groups.append((positive, *selected))
    if not groups:
        raise RuntimeError(f"no listwise groups with {negatives_per_group} unique negatives in {path}")
    return groups


class ListwiseHardDataset(Dataset):
    def __init__(self, path: Path, negatives_per_group: int = 2):
        self.groups = build_hard_groups(path, negatives_per_group)
        self.group_size = negatives_per_group + 1

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):
        return {"rows": self.groups[index]}


class PairwiseCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        ordered = [row[key] for row in rows for key in ("positive", "negative")]
        children = [render_role_text(row["child"], row.get("child_gloss", ""), "child") for row in ordered]
        parents = [render_role_text(row["parent"], row.get("parent_gloss", ""), "parent") for row in ordered]
        inputs = self.tokenizer(children, parents, padding=True, truncation=True,
                                max_length=self.max_length, return_tensors="pt")
        return {"pair_inputs": inputs, "pair_count": len(rows)}


class ListwiseCollator:
    def __init__(self, tokenizer, max_length: int, group_size: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.group_size = group_size

    def __call__(self, rows):
        ordered = [pair for row in rows for pair in row["rows"]]
        if any(len(row["rows"]) != self.group_size for row in rows):
            raise ValueError("listwise batch contains inconsistent group sizes")
        children = [render_role_text(row["child"], row.get("child_gloss", ""), "child") for row in ordered]
        parents = [render_role_text(row["parent"], row.get("parent_gloss", ""), "parent") for row in ordered]
        inputs = self.tokenizer(
            children,
            parents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"pair_inputs": inputs, "group_count": len(rows), "group_size": self.group_size}


def _direct_score(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 0].float() - logits[:, 1].float()


@torch.inference_mode()
def evaluate_pairwise(model, loader, device, precision):
    model.eval()
    total_loss = total = correct = 0
    for step, batch in enumerate(loader):
        inputs = {key: value.to(device) for key, value in batch["pair_inputs"].items()}
        with autocast_context(device, precision):
            logits = model(inputs)["relation_logits"]
        scores = _direct_score(logits).reshape(-1, 2)
        margins = scores[:, 0] - scores[:, 1]
        loss = nn.functional.softplus(-margins)
        require_finite(loss, "pairwise dev loss", epoch=0, step=step)
        total_loss += float(loss.sum())
        total += len(margins)
        correct += int((margins > 0).sum())
    return {"loss": total_loss / max(1, total), "accuracy": correct / max(1, total),
            "macro_f1": correct / max(1, total), "pairwise_accuracy": correct / max(1, total)}


@torch.inference_mode()
def evaluate_listwise(model, loader, device, precision):
    model.eval()
    total_loss = total = correct = 0
    for step, batch in enumerate(loader):
        inputs = {key: value.to(device) for key, value in batch["pair_inputs"].items()}
        with autocast_context(device, precision):
            logits = model(inputs)["relation_logits"]
        scores = _direct_score(logits).reshape(-1, batch["group_size"])
        targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        loss = nn.functional.cross_entropy(scores, targets, reduction="none")
        require_finite(loss, "listwise dev loss", epoch=0, step=step)
        total_loss += float(loss.sum())
        total += len(loss)
        correct += int((scores.argmax(dim=1) == 0).sum())
    accuracy = correct / max(1, total)
    return {"loss": total_loss / max(1, total), "accuracy": accuracy,
            "macro_f1": accuracy, "listwise_top1_accuracy": accuracy}


def parse_args():
    base = Path("data/processed/taxobranch/wordnet_relations")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=base / "relations/train.jsonl")
    parser.add_argument("--dev", type=Path, default=base / "relations/dev.jsonl")
    parser.add_argument("--train-roots", type=Path, default=base / "roots/train.jsonl")
    parser.add_argument("--dev-roots", type=Path, default=base / "roots/dev.jsonl")
    parser.add_argument("--objective", choices=("balanced_binary", "pairwise", "listwise"), required=True)
    parser.add_argument("--listwise-negatives", type=int, default=2)
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--root-batch-size", type=int, default=64)
    parser.add_argument(
        "--max-root-batches-per-step",
        type=int,
        default=0,
        help="Cap auxiliary ROOT batches per relation step; 0 preserves a full ROOT pass.",
    )
    parser.add_argument("--examples-per-epoch", type=int, default=399228,
                        help="Pair encodings per epoch; pairwise batches consume two encodings per item")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--root-weight", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--precision", choices=PRECISIONS, default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_root_batches_per_step < 0:
        raise ValueError("max-root-batches-per-step must be non-negative")
    seed_everything(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    if args.objective == "balanced_binary":
        train_set, dev_set = BalancedHardDataset(args.train), BalancedHardDataset(args.dev)
        collator = RelationCollator(tokenizer, args.max_length)
        samples_per_epoch = args.examples_per_epoch
        evaluator = evaluate
    elif args.objective == "pairwise":
        train_set, dev_set = PairwiseHardDataset(args.train), PairwiseHardDataset(args.dev)
        collator = PairwiseCollator(tokenizer, args.max_length)
        samples_per_epoch = math.ceil(args.examples_per_epoch / 2)
        evaluator = evaluate_pairwise
    else:
        train_set = ListwiseHardDataset(args.train, args.listwise_negatives)
        dev_set = ListwiseHardDataset(args.dev, args.listwise_negatives)
        group_size = args.listwise_negatives + 1
        if args.examples_per_epoch % group_size:
            raise ValueError("examples-per-epoch must be divisible by listwise group size")
        collator = ListwiseCollator(tokenizer, args.max_length, group_size)
        samples_per_epoch = args.examples_per_epoch // group_size
        evaluator = evaluate_listwise
    root_train, root_dev = RootJsonlDataset(args.train_roots), RootJsonlDataset(args.dev_roots)
    loader_kw = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda",
                 "persistent_workers": args.num_workers > 0}
    if args.num_workers:
        loader_kw["multiprocessing_context"] = "spawn"
    generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(train_set, replacement=True, num_samples=samples_per_epoch, generator=generator)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collator, **loader_kw)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size * 2, collate_fn=collator, **loader_kw)
    root_collator = RootCollator(tokenizer, args.max_length)
    root_loader = DataLoader(root_train, batch_size=args.root_batch_size, shuffle=True, collate_fn=root_collator, **loader_kw)
    root_dev_loader = DataLoader(root_dev, batch_size=args.root_batch_size * 2, collate_fn=root_collator, **loader_kw)
    model = RelationCrossEncoder(RelationScorerConfig(encoder_name=args.encoder, relation_objective="binary")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_updates = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_updates * args.warmup_ratio), total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=uses_fp16(device, args.precision))
    root_counts = Counter(int(row["is_root"]) for row in root_train.rows)
    root_pos_weight = torch.tensor(root_counts[0] / max(1, root_counts[1]), device=device)
    auxiliary_steps = len(root_loader)
    if args.max_root_batches_per_step:
        auxiliary_steps = min(
            auxiliary_steps,
            len(train_loader) * args.max_root_batches_per_step,
        )
    root_frequency_scale = len(train_loader) / auxiliary_steps
    early = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)
    args.out.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "relation_objective_saved": "binary", "hardness_order": HARDNESS,
              "fixed_negative_construction": True, "train_pairs": len(build_hard_pairs(args.train)),
              "dev_pairs": len(build_hard_pairs(args.dev)), "samples_per_epoch": samples_per_epoch,
              "pair_encodings_per_epoch": samples_per_epoch * (
                  2 if args.objective == "pairwise"
                  else args.listwise_negatives + 1 if args.objective == "listwise"
                  else 1
              ),
              "optimizer_updates_per_epoch": len(train_loader), "root_steps_per_epoch": auxiliary_steps,
              "resolved_precision": str(resolve_precision(device, args.precision))}
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    history, started = [], time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        root_iterator = iter(root_loader)
        relation_sum = root_sum = 0.0
        seen = root_seen = 0
        for step, batch in enumerate(train_loader):
            pair = {key: value.to(device, non_blocking=True) for key, value in batch["pair_inputs"].items()}
            with autocast_context(device, args.precision):
                logits = model(pair)["relation_logits"]
                if args.objective == "balanced_binary":
                    labels = batch["labels"].to(device, non_blocking=True)
                    relation_loss = nn.functional.cross_entropy(logits.float(), labels)
                    batch_seen = labels.numel()
                elif args.objective == "pairwise":
                    scores = _direct_score(logits).reshape(-1, 2)
                    relation_loss = nn.functional.softplus(-(scores[:, 0] - scores[:, 1])).mean()
                    batch_seen = scores.numel()
                else:
                    scores = _direct_score(logits).reshape(-1, batch["group_size"])
                    targets = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
                    relation_loss = nn.functional.cross_entropy(scores, targets)
                    batch_seen = scores.numel()
                loss = relation_loss
                for _ in range(auxiliary_batches_for_step(step, len(train_loader), auxiliary_steps)):
                    try:
                        root_batch = next(root_iterator)
                    except StopIteration:
                        root_iterator = iter(root_loader); root_batch = next(root_iterator)
                    roots = {key: value.to(device, non_blocking=True) for key, value in root_batch["inputs"].items()}
                    root_labels = root_batch["labels"].to(device, non_blocking=True)
                    root_logits = model.root_head(model.encode(roots)).squeeze(-1)
                    root_loss = nn.functional.binary_cross_entropy_with_logits(root_logits.float(), root_labels,
                                                                                pos_weight=root_pos_weight)
                    loss = loss + args.root_weight * root_frequency_scale * root_loss
                    root_sum += float(root_loss.detach()) * root_labels.numel(); root_seen += root_labels.numel()
            require_finite(loss, "matched control loss", epoch=epoch, step=step)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            require_finite(grad_norm, "gradient norm", epoch=epoch, step=step)
            scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            relation_sum += float(relation_loss.detach()) * batch_seen; seen += batch_seen
        metrics = evaluator(model, dev_loader, device, args.precision)
        metrics.update(evaluate_roots(model, root_dev_loader, device, args.precision))
        metrics.update({"epoch": epoch, "train_loss": relation_sum / max(1, seen),
                        "train_root_loss": root_sum / max(1, root_seen), "pair_encodings_seen": seen})
        improved, stop = early.update(metrics["macro_f1"], epoch)
        metrics.update({"improved": improved, "best_epoch": early.best_epoch, "best_macro_f1": early.best})
        history.append(metrics); print(json.dumps(metrics, sort_keys=True), flush=True)
        if improved:
            model.save_pretrained(args.out / "best"); tokenizer.save_pretrained(args.out / "best/tokenizer")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stop:
            break
    (args.out / "training_summary.json").write_text(json.dumps({
        "objective": args.objective, "epochs_completed": len(history), "early_stopping": early.as_dict(),
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "pair_encodings_per_epoch": config["pair_encodings_per_epoch"], "device": str(device),
    }, indent=2) + "\n")
    if not (args.out / "best/relation_heads.pt").exists():
        raise RuntimeError("training produced no best checkpoint")


if __name__ == "__main__":
    main()
