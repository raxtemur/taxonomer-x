"""Listwise scoring of ROOT, singleton, and unordered parent-pair sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
import random

import numpy as np
import torch
from torch import nn

from src.models.multiparent_evidence import _pair_features, _ranked_rows
from src.symbolic.multiparent_projection import DagCandidate


class JointParentSetMLP(nn.Module):
    """One normalized candidate-set space with a symmetric pair branch."""

    def __init__(self, edge_dim: int = 768, hidden_dim: int = 96):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden_dim), nn.GELU())
        self.single = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.pair = nn.Sequential(
            nn.Linear(3 * hidden_dim + 8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.root = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_repr, single_scalar, valid, pair_indices, pair_scalar, root_raw):
        h = self.edge(edge_repr.float())
        single = self.single(torch.cat((h, single_scalar.float()), dim=-1)).squeeze(-1)
        single = single.masked_fill(~valid, -1e9)
        weights = valid.float().unsqueeze(-1)
        pooled = (h * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        root = self.root(torch.cat((pooled, root_raw.float().unsqueeze(-1)), dim=-1)).squeeze(-1)
        left, right = pair_indices[:, 0], pair_indices[:, 1]
        a, b = h[:, left], h[:, right]
        pair_valid = valid[:, left] & valid[:, right]
        symmetric = torch.cat((a + b, torch.abs(a - b), a * b, pair_scalar.float()), dim=-1)
        pair = self.pair(symmetric).squeeze(-1).masked_fill(~pair_valid, -1e9)
        return torch.cat((root[:, None], single, pair), dim=1)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _indexed_children(records, score_maps, repr_maps, roots, edge_calibration, k, q):
    examples = []
    pair_positions = list(combinations(range(q), 2))
    for record, score_map, root_logits in zip(records, score_maps, roots):
        gold = {}
        for child, parent in record.gold:
            gold.setdefault(child, set()).add(parent)
        for child, node in enumerate(record.nodes):
            rows = _ranked_rows(record, score_map, child, k)
            if not rows:
                continue
            calibrated = edge_calibration.log_odds([[r["raw_score"]] for r in rows]).tolist()
            ranked = sorted(zip(rows, calibrated), key=lambda item: (-item[1], item[0]["parent"]))
            rows, calibrated = zip(*ranked)
            parents = [row["parent"] for row in rows]
            target_parents = gold.get(node, set())
            if len(target_parents) == 0:
                target = 0
            elif len(target_parents) == 1 and next(iter(target_parents)) in parents:
                target = 1 + parents.index(next(iter(target_parents)))
            elif len(target_parents) == 2 and target_parents <= set(parents[:q]):
                indices = tuple(sorted(parents.index(parent) for parent in target_parents))
                target = 1 + k + pair_positions.index(indices)
            else:
                continue
            examples.append((record, child, list(rows), list(calibrated), float(root_logits[child]), target))
    return examples


def _batch(examples, repr_maps, k, q, device):
    pair_positions = list(combinations(range(q), 2))
    edge_dim = next(iter(repr_maps[examples[0][0].key].values())).numel()
    b = len(examples)
    representations = torch.zeros(b, k, edge_dim, dtype=torch.float16)
    single_scalar = torch.zeros(b, k, 3)
    valid = torch.zeros(b, k, dtype=torch.bool)
    pair_scalar = torch.zeros(b, len(pair_positions), 8)
    roots, targets = [], []
    for bi, (record, child, rows, logits, raw_root, target) in enumerate(examples):
        mapping = repr_maps[record.key]
        n = min(k, len(rows))
        valid[bi, :n] = True
        roots.append(raw_root); targets.append(target)
        for index, (row, logit) in enumerate(zip(rows[:n], logits[:n])):
            representations[bi, index] = mapping[(child, row["rank"])].half()
            single_scalar[bi, index] = torch.tensor(
                [float(logit), float(row["retrieval_score"]), math.log1p(index)], dtype=torch.float32
            )
        for pi, (left, right) in enumerate(pair_positions):
            if right < n:
                pair_scalar[bi, pi] = torch.tensor(
                    _pair_features(rows[left], rows[right], logits[left], logits[right]), dtype=torch.float32
                )
    return (
        representations.to(device), single_scalar.to(device), valid.to(device),
        torch.tensor(pair_positions, dtype=torch.long, device=device), pair_scalar.to(device),
        torch.tensor(roots, dtype=torch.float32, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def fit_joint_parent_set_model(
    records, score_maps, repr_maps, roots, edge_calibration, *, k=50, q=10,
    hidden_dim=96, epochs=12, batch_size=32, lr=1e-3, weight_decay=1e-4,
    seed=0, device="auto",
):
    torch.manual_seed(seed); random.seed(seed)
    target_device = _device(device)
    examples = _indexed_children(records, score_maps, repr_maps, roots, edge_calibration, k, q)
    if not examples:
        raise ValueError("joint set training has no fully retrieved exact-set examples")
    edge_dim = next(iter(repr_maps[examples[0][0].key].values())).numel()
    model = JointParentSetMLP(edge_dim, hidden_dim).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(examples), generator=generator).tolist()
        model.train()
        for start in range(0, len(order), batch_size):
            batch = _batch([examples[i] for i in order[start:start + batch_size]], repr_maps, k, q, target_device)
            logits = model(*batch[:-1])
            loss = nn.functional.cross_entropy(logits, batch[-1])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
    model.eval()
    types = {"root": 0, "singleton": 0, "pair": 0}
    for example in examples:
        target = example[-1]
        types["root" if target == 0 else "singleton" if target <= k else "pair"] += 1
    return model, {
        "objective": "single_listwise_softmax_over_root_singletons_unordered_pairs",
        "examples": len(examples), "target_types": types, "epochs": epochs,
        "train_q": q, "candidate_k": k, "final_loss": losses[-1],
        "device": str(target_device), "pair_symmetry": "by_construction",
    }


@dataclass
class JointSetEvidence:
    base: object
    model: JointParentSetMLP
    repr_maps: dict
    mode: str = "fixed"
    fixed_q: int = 5
    thresholds: tuple[float, float] | None = None
    pair_m: int = 1
    inference_batch_size: int = 256
    _cache: dict = field(default_factory=dict, repr=False)
    _score_cache: dict = field(default_factory=dict, repr=False)
    _resource_audit: dict = field(default_factory=dict, repr=False)

    @property
    def edge(self):
        return self.base.edge

    def _score_children(self, record, score_map, raw_roots, children, q):
        """Score many children in one GPU call and share results across adapters."""
        children = tuple(int(child) for child in children)
        q = min(int(q), int(self.base.max_k))
        if q < 2 or self.inference_batch_size < 1:
            raise ValueError("q must be >=2 and inference_batch_size must be positive")
        keys = {
            child: (record.key, id(score_map), q, child, float(raw_roots[child]))
            for child in children
        }
        missing = [child for child in children if keys[child] not in self._score_cache]
        audit = self._resource_audit
        audit["score_requests"] = audit.get("score_requests", 0) + len(children)
        audit["score_cache_hits"] = (
            audit.get("score_cache_hits", 0) + len(children) - len(missing)
        )

        prepared = {}
        for child in missing:
            rows = _ranked_rows(record, score_map, child, self.base.max_k)
            calibrated = (
                self.base.edge.log_odds([[r["raw_score"]] for r in rows]).tolist()
                if rows else []
            )
            ranked = sorted(
                zip(rows, calibrated),
                key=lambda item: (-item[1], item[0]["parent"]),
            )
            rows = [item[0] for item in ranked]
            calibrated = [float(item[1]) for item in ranked]
            if not rows:
                self._score_cache[keys[child]] = (
                    rows, calibrated, float(raw_roots[child]), [], []
                )
            else:
                prepared[child] = (rows, calibrated)

        pending = list(prepared)
        mapping = self.repr_maps[record.key]
        edge_dim = next(iter(mapping.values())).numel() if pending else 0
        device = next(self.model.parameters()).device
        pair_positions = list(combinations(range(q), 2))
        pair_index = torch.tensor(pair_positions, dtype=torch.long, device=device)
        for start in range(0, len(pending), self.inference_batch_size):
            chunk = pending[start:start + self.inference_batch_size]
            b = len(chunk)
            k = int(self.base.max_k)
            representation = torch.zeros(b, k, edge_dim, dtype=torch.float16)
            scalar = torch.zeros(b, k, 3)
            valid = torch.zeros(b, k, dtype=torch.bool)
            pair_scalar = torch.zeros(b, len(pair_positions), 8)
            batch_roots = torch.empty(b, dtype=torch.float32)
            for bi, child in enumerate(chunk):
                rows, calibrated = prepared[child]
                n = min(k, len(rows))
                valid[bi, :n] = True
                batch_roots[bi] = float(raw_roots[child])
                for index, (row, logit) in enumerate(zip(rows[:n], calibrated[:n])):
                    representation[bi, index] = mapping[(child, row["rank"])]
                    scalar[bi, index] = torch.tensor(
                        [logit, row["retrieval_score"], math.log1p(index)]
                    )
                for pi, (left, right) in enumerate(pair_positions):
                    if right < n:
                        pair_scalar[bi, pi] = torch.tensor(
                            _pair_features(
                                rows[left], rows[right],
                                calibrated[left], calibrated[right],
                            )
                        )
            with torch.inference_mode():
                output = self.model(
                    representation.to(device), scalar.to(device), valid.to(device),
                    pair_index, pair_scalar.to(device), batch_roots.to(device),
                ).cpu()
            for bi, child in enumerate(chunk):
                rows, calibrated = prepared[child]
                effective_q = min(q, len(rows))
                pair_count = effective_q * (effective_q - 1) // 2
                logits = output[bi].tolist()
                self._score_cache[keys[child]] = (
                    rows, calibrated, float(logits[0]),
                    [float(value) for value in logits[1:1 + k]],
                    [float(value) for value in logits[1 + k:1 + k + pair_count]],
                )
            audit["gpu_batches"] = audit.get("gpu_batches", 0) + 1
            audit["gpu_scored_children"] = audit.get("gpu_scored_children", 0) + b
        return {child: self._score_cache[keys[child]] for child in children}

    def _score_child(self, record, score_map, raw_root, child, q):
        roots = [0.0] * len(record.nodes)
        roots[child] = float(raw_root)
        return self._score_children(record, score_map, roots, (child,), q)[child]

    def _adaptive_qs(self, record, score_map, raw_roots):
        """Allocate a graph-level uncertainty frontier under a fixed q=5 budget."""
        children = range(len(record.nodes))
        scored3 = self._score_children(record, score_map, raw_roots, children, 3)
        gaps3 = []
        for child in children:
            *_, root, singles, pairs = scored3[child]
            values = sorted([root, *singles, *pairs], reverse=True)
            gaps3.append(values[0] - values[1] if len(values) > 1 else float("inf"))
        expand5 = max(1, round(0.40 * len(record.nodes))) if record.nodes else 0
        uncertain = sorted(range(len(record.nodes)), key=lambda child: (gaps3[child], child))[:expand5]
        gaps5 = []
        scored5 = self._score_children(record, score_map, raw_roots, uncertain, 5)
        for child in uncertain:
            *_, root, singles, pairs = scored5[child]
            values = sorted([root, *singles, *pairs], reverse=True)
            gaps5.append((values[0] - values[1] if len(values) > 1 else float("inf"), child))
        uncertain10 = {child for _, child in sorted(gaps5)[:round(0.25 * len(uncertain))]}
        uncertain5 = set(uncertain)
        return [10 if child in uncertain10 else 5 if child in uncertain5 else 3 for child in range(len(record.nodes))]

    def transform_with_roots(self, record, score_map, raw_roots):
        key = (record.key, self.mode, self.fixed_q, self.thresholds)
        if key in self._cache:
            return self._cache[key]
        candidates, pair_scores, allowed, cardinality, root_scores = [], {}, set(), {}, {}
        q_counts = {3: 0, 5: 0, 10: 0}
        q_by_child = (
            self._adaptive_qs(record, score_map, raw_roots)
            if self.mode == "adaptive"
            else [self.fixed_q] * len(record.nodes)
        )
        by_q = {}
        for child, q in enumerate(q_by_child):
            by_q.setdefault(q, []).append(child)
        for q, children in by_q.items():
            self._score_children(record, score_map, raw_roots, children, q)
        for child, node in enumerate(record.nodes):
            q = q_by_child[child]
            q_counts[q] = q_counts.get(q, 0) + 1
            rows, _, root, singles, pairs = self._score_child(record, score_map, raw_roots[child], child, q)
            root_scores[node] = root; cardinality[node] = {0: 0.0, 1: 0.0, 2: 0.0}
            for row, value in zip(rows, singles):
                candidates.append(DagCandidate(node, row["parent"], value))
            scored = []
            for value, (left, right) in zip(pairs, combinations(range(min(q, len(rows))), 2)):
                scored.append((value, left, right))
            scored.sort(key=lambda item: (-item[0], rows[item[1]]["parent"], rows[item[2]]["parent"]))
            for value, left, right in scored[:self.pair_m]:
                left_parent, right_parent = rows[left]["parent"], rows[right]["parent"]
                pair_scores[(node, *sorted((left_parent, right_parent)))] = value - singles[left] - singles[right]
                allowed.update(((node, left_parent), (node, right_parent)))
        result = {
            "candidates": candidates, "cardinality_scores": cardinality,
            "pair_scores": pair_scores, "multi_parent_candidate_edges": allowed,
            "root_scores": root_scores, "q_counts": q_counts,
            "pair_evaluations": sum(count * (q * (q - 1) // 2) for q, count in q_counts.items()),
        }
        self._cache[key] = result
        return result

    def transform(self, record, score_map):
        raise RuntimeError("JointSetEvidence requires raw root logits")

    def ranking_audit(self, record, score_map, raw_roots):
        transformed = self.transform_with_roots(record, score_map, raw_roots)
        gold = {}
        for child, parent in record.gold:
            gold.setdefault(child, set()).add(parent)
        eligible = hit = 0
        for child, left, right in transformed["pair_scores"]:
            if len(gold.get(child, set())) == 2:
                available = {record.nodes[p] for p, _ in record.retrieved[record.nodes.index(child)][:self.base.max_k]}
                if gold[child] <= available:
                    eligible += 1; hit += int({left, right} == gold[child])
        return {"eligible_exact_pairs": eligible, "gold_pair_in_top_m": hit, "gold_pair_recall_at_m": hit / eligible if eligible else 0.0}


def fit_adaptive_thresholds(adapter, records, score_maps, roots):
    """Freeze uncertainty cutoffs with a pair cost no larger than fixed q=5."""
    gaps3, gaps3_by_record = [], []
    for record, score_map, raw_roots in zip(records, score_maps, roots):
        scored = adapter._score_children(
            record, score_map, raw_roots, range(len(record.nodes)), 3
        )
        record_gaps = []
        for child in range(len(record.nodes)):
            *_, root, singles, pairs = scored[child]
            values = sorted([root, *singles, *pairs], reverse=True)
            gap = values[0] - values[1] if len(values) > 1 else float("inf")
            gaps3.append(gap)
            record_gaps.append(gap)
        gaps3_by_record.append(record_gaps)
    threshold3 = float(np.quantile(gaps3, 0.40))
    gaps5 = []
    for record, score_map, raw_roots, record_gaps in zip(
        records, score_maps, roots, gaps3_by_record
    ):
        selected = [child for child, gap in enumerate(record_gaps) if gap < threshold3]
        scored = adapter._score_children(record, score_map, raw_roots, selected, 5)
        for child in selected:
            *_, root, singles, pairs = scored[child]
            values = sorted([root, *singles, *pairs], reverse=True)
            gaps5.append(values[0] - values[1] if len(values) > 1 else float("inf"))
    threshold5 = float(np.quantile(gaps5, 0.25)) if gaps5 else threshold3
    expected_pair_cost = 0.60 * 3 + 0.30 * 10 + 0.10 * 45
    if expected_pair_cost > 10.0:
        raise AssertionError("adaptive frontier exceeds fixed-q5 pair budget")
    return (threshold3, threshold5), {
        "selection_labels_used": False, "q3_expand_fraction": 0.40,
        "expanded_q5_expand_fraction": 0.25, "expected_pairs_per_child": expected_pair_cost,
        "fixed_q5_pairs_per_child": 10,
    }
