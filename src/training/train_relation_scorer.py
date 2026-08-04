"""Resource-efficient five-way relation and uniformly weighted ROOT training."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.relation_aware import RELATION_OBJECTIVES, RelationCrossEncoder, RelationScorerConfig, target_for_objective
from src.training.relation_dataset import RelationCollator, RelationJsonlDataset
from src.training.root_dataset import RootCollator, RootJsonlDataset
from src.training.early_stopping import EarlyStopping
from src.utils.precision import PRECISIONS, autocast_context, resolve_precision, uses_fp16


def auxiliary_batches_for_step(step: int, primary_steps: int, auxiliary_steps: int) -> int:
    """Evenly interleave exactly `auxiliary_steps` over one primary epoch."""
    return ((step + 1) * auxiliary_steps) // primary_steps - (step * auxiliary_steps) // primary_steps


def require_finite(value: torch.Tensor, label: str, *, epoch: int, step: int) -> None:
    """Abort at the first numerical fault instead of saving a poisoned checkpoint."""
    if not bool(torch.isfinite(value.detach()).all()):
        raise FloatingPointError(f"Non-finite {label} at epoch={epoch} step={step}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader, device, precision="auto"):
    model.eval()
    relation_labels = model.relation_labels
    confusion = torch.zeros((len(relation_labels), len(relation_labels)), dtype=torch.long)
    total_loss = 0.0
    total = 0
    for batch in loader:
        pair = {k: v.to(device) for k, v in batch["pair_inputs"].items()}
        labels = batch["labels"].to(device)
        with autocast_context(device, precision):
            logits = model(pair)["relation_logits"]
        logits = logits.float()
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        require_finite(loss, "dev relation loss", epoch=0, step=total)
        total_loss += float(loss)
        total += labels.numel()
        for gold, pred in zip(labels.cpu(), logits.argmax(-1).cpu()):
            confusion[gold, pred] += 1
    f1s = []
    for i in range(len(relation_labels)):
        tp = int(confusion[i, i])
        fp = int(confusion[:, i].sum()) - tp
        fn = int(confusion[i, :].sum()) - tp
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {
        "loss": total_loss / max(1, total),
        "accuracy": float(confusion.diag().sum()) / max(1, int(confusion.sum())),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class_f1": dict(zip(relation_labels, f1s)),
        "confusion": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate_roots(model, loader, device, precision="auto"):
    model.eval()
    tp = fp = fn = total = correct = 0
    loss = 0.0
    for batch in loader:
        inputs = {k: v.to(device) for k, v in batch["inputs"].items()}
        labels = batch["labels"].to(device)
        with autocast_context(device, precision):
            logits = model.root_head(model.encode(inputs)).squeeze(-1)
        logits = logits.float()
        batch_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
        require_finite(batch_loss, "dev root loss", epoch=0, step=total)
        loss += float(batch_loss)
        pred = logits >= 0
        gold = labels > 0.5
        tp += int((pred & gold).sum())
        fp += int((pred & ~gold).sum())
        fn += int((~pred & gold).sum())
        correct += int((pred == gold).sum())
        total += labels.numel()
    return {
        "root_loss": loss / max(1, total),
        "root_accuracy": correct / max(1, total),
        "root_f1": 2 * tp / max(1, 2 * tp + fp + fn),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/processed/taxobranch/wordnet_relations")
    parser.add_argument("--train", type=Path, default=base / "relations/train.jsonl")
    parser.add_argument("--dev", type=Path, default=base / "relations/dev.jsonl")
    parser.add_argument("--train-roots", type=Path, default=base / "roots/train.jsonl")
    parser.add_argument("--dev-roots", type=Path, default=base / "roots/dev.jsonl")
    parser.add_argument("--encoder", default="microsoft/deberta-v3-base")
    parser.add_argument("--relation-objective", choices=RELATION_OBJECTIVES, default="typed",
                        help="typed five-way objective or matched binary direct/non-direct control")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--root-batch-size", type=int, default=64)
    parser.add_argument("--root-passes-per-epoch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--root-weight", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--precision", choices=PRECISIONS, default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--early-stopping-patience", type=int, default=2,
                        help="Stop after this many dev macro-F1 plateaus; 0 disables")
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.root_passes_per_epoch < 1 or args.gradient_accumulation < 1:
        raise ValueError("passes and accumulation must be positive")
    if args.batch_size < 1 or args.root_batch_size < 1 or not 0 <= args.warmup_ratio < 1:
        raise ValueError("invalid batch size or warmup ratio")
    early_stopping = EarlyStopping(
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    pair_collator = RelationCollator(tokenizer, args.max_length)
    root_collator = RootCollator(tokenizer, args.max_length)
    train_set = RelationJsonlDataset(args.train, relation_objective=args.relation_objective)
    dev_set = RelationJsonlDataset(args.dev, relation_objective=args.relation_objective)
    root_train = RootJsonlDataset(args.train_roots)
    root_dev = RootJsonlDataset(args.dev_roots)
    loader_kw = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        # Forking DataLoader workers after CUDA initialization can hang a job
        # before its first batch. Spawn gives every worker a clean interpreter.
        loader_kw["multiprocessing_context"] = "spawn"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=pair_collator, **loader_kw)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size * 2, collate_fn=pair_collator, **loader_kw)
    root_loader = DataLoader(root_train, batch_size=args.root_batch_size, shuffle=True, collate_fn=root_collator, **loader_kw)
    root_dev_loader = DataLoader(root_dev, batch_size=args.root_batch_size * 2, collate_fn=root_collator, **loader_kw)

    model = RelationCrossEncoder(RelationScorerConfig(
        encoder_name=args.encoder, relation_objective=args.relation_objective
    )).to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("RelationCrossEncoder optimizer parameters must use FP32 master weights")
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        model.encoder.config.use_cache = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation)
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)
    scaler = torch.amp.GradScaler("cuda", enabled=uses_fp16(device, args.precision))

    counts = Counter(target_for_objective(row["relation"], args.relation_objective) for row in train_set.rows)
    weights = torch.tensor(
        [len(train_set) / max(1, len(model.relation_labels) * counts[label]) for label in model.relation_labels],
        dtype=torch.float32,
        device=device,
    )
    root_counts = Counter(int(row["is_root"]) for row in root_train.rows)
    root_pos_weight = torch.tensor(root_counts[0] / max(1, root_counts[1]), dtype=torch.float32, device=device)
    primary_steps = len(train_loader)
    auxiliary_steps = len(root_loader) * args.root_passes_per_epoch
    root_frequency_scale = primary_steps / auxiliary_steps
    resolved = resolve_precision(device, args.precision)

    history = []
    training_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    args.out.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "train_examples": len(train_set),
        "dev_examples": len(dev_set),
        "root_train_examples": len(root_train),
        "primary_steps_per_epoch": primary_steps,
        "root_steps_per_epoch": auxiliary_steps,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "root_frequency_scale": root_frequency_scale,
        "resolved_precision": str(resolved or torch.float32),
        "master_weight_dtype": str(next(model.parameters()).dtype),
    }
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in run_config.items()}
    (args.out / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps({"training_config": run_config}, sort_keys=True), flush=True)

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        relation_sum = root_sum = 0.0
        seen = root_seen = 0
        max_grad_norm = 0.0
        root_iterator = iter(root_loader)
        for step, batch in enumerate(train_loader):
            pair = {k: v.to(device, non_blocking=True) for k, v in batch["pair_inputs"].items()}
            labels = batch["labels"].to(device, non_blocking=True)
            with autocast_context(device, args.precision):
                relation_logits = model(pair)["relation_logits"]
                relation_loss = nn.functional.cross_entropy(relation_logits.float(), labels, weight=weights)
                loss = relation_loss
                for _ in range(auxiliary_batches_for_step(step, primary_steps, auxiliary_steps)):
                    try:
                        root_batch = next(root_iterator)
                    except StopIteration:
                        root_iterator = iter(root_loader)
                        root_batch = next(root_iterator)
                    roots = {k: v.to(device, non_blocking=True) for k, v in root_batch["inputs"].items()}
                    root_labels = root_batch["labels"].to(device, non_blocking=True)
                    root_logits = model.root_head(model.encode(roots)).squeeze(-1)
                    root_loss = nn.functional.binary_cross_entropy_with_logits(
                        root_logits.float(), root_labels, pos_weight=root_pos_weight
                    )
                    require_finite(root_loss, "root loss", epoch=epoch, step=step)
                    loss = loss + args.root_weight * root_frequency_scale * root_loss
                    root_sum += float(root_loss.detach()) * root_labels.numel()
                    root_seen += root_labels.numel()
                loss = loss / args.gradient_accumulation
            require_finite(relation_loss, "relation loss", epoch=epoch, step=step)
            require_finite(loss, "combined loss", epoch=epoch, step=step)
            scaler.scale(loss).backward()
            should_step = (step + 1) % args.gradient_accumulation == 0 or step + 1 == primary_steps
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                require_finite(grad_norm, "gradient norm", epoch=epoch, step=step)
                max_grad_norm = max(max_grad_norm, float(grad_norm))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            relation_sum += float(relation_loss.detach()) * labels.numel()
            seen += labels.numel()
            if args.log_every and (step + 1) % args.log_every == 0:
                print(json.dumps({
                    "epoch": epoch,
                    "step": step + 1,
                    "steps": primary_steps,
                    "relation_loss": relation_sum / max(1, seen),
                    "root_loss": root_sum / max(1, root_seen),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "max_grad_norm": max_grad_norm,
                }, sort_keys=True), flush=True)

        metrics = evaluate(model, dev_loader, device, args.precision)
        metrics.update(evaluate_roots(model, root_dev_loader, device, args.precision))
        metrics.update(
            epoch=epoch,
            train_loss=relation_sum / max(1, seen),
            train_root_loss=root_sum / max(1, root_seen),
            root_pos_weight=float(root_pos_weight),
            root_examples_seen=root_seen,
            root_frequency_scale=root_frequency_scale,
            precision=args.precision,
            resolved_precision=str(resolved or torch.float32),
            master_weight_dtype=str(next(model.parameters()).dtype),
            gradient_accumulation=args.gradient_accumulation,
            batch_size=args.batch_size,
            optimizer_updates=updates_per_epoch,
            max_grad_norm=max_grad_norm,
            epoch_wall_seconds=time.perf_counter() - epoch_started,
            relation_examples_per_second=seen / max(time.perf_counter() - epoch_started, 1e-9),
            peak_cuda_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        )
        if not math.isfinite(metrics["macro_f1"]):
            raise FloatingPointError(f"Non-finite dev macro-F1 at epoch={epoch}")
        improved, should_stop = early_stopping.update(metrics["macro_f1"], epoch)
        metrics.update(
            improved=improved,
            best_macro_f1=early_stopping.best,
            best_epoch=early_stopping.best_epoch,
            epochs_without_improvement=early_stopping.bad_epochs,
        )
        history.append(metrics)
        print(json.dumps(metrics, sort_keys=True), flush=True)
        if improved:
            model.save_pretrained(args.out / "best")
            tokenizer.save_pretrained(args.out / "best/tokenizer")
        (args.out / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        if should_stop:
            print(json.dumps({"early_stopping": early_stopping.as_dict()}), flush=True)
            break

    (args.out / "training_summary.json").write_text(json.dumps({
        "epochs_completed": len(history),
        "epochs_requested": args.epochs,
        "early_stopping": early_stopping.as_dict(),
        "relation_objective": args.relation_objective,
        "relation_labels": list(model.relation_labels),
        "total_wall_seconds": time.perf_counter() - training_started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }, indent=2) + "\n")
    if early_stopping.best_epoch == 0 or not (args.out / "best/relation_heads.pt").exists():
        raise RuntimeError("Training completed without a finite best checkpoint")


if __name__ == "__main__":
    main()
