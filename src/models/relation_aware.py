"""Relation- and candidate-graph-aware neural edge potentials."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


TYPED_RELATION_LABELS = ("direct", "ancestor", "sibling", "reverse", "unrelated")
BINARY_RELATION_LABELS = ("direct", "non_direct")
RELATION_OBJECTIVES = ("typed", "binary")
# Backwards-compatible public names used by historical checkpoints and analyses.
RELATION_LABELS = TYPED_RELATION_LABELS
RELATION_TO_ID = {label: index for index, label in enumerate(RELATION_LABELS)}


def labels_for_objective(objective: str) -> tuple[str, ...]:
    if objective == "typed":
        return TYPED_RELATION_LABELS
    if objective == "binary":
        return BINARY_RELATION_LABELS
    raise ValueError(f"Unknown relation objective: {objective}")


def target_for_objective(relation: str, objective: str) -> str:
    return relation if objective == "typed" or relation == "direct" else "non_direct"


@dataclass
class RelationScorerConfig:
    encoder_name: str = "microsoft/deberta-v3-base"
    dropout: float = 0.1
    ancestor_penalty: float = 0.5
    reverse_penalty: float = 1.0
    sibling_penalty: float = 0.25
    unrelated_penalty: float = 0.25
    relation_objective: str = "typed"


def render_role_text(term: str, gloss: str, role: str) -> str:
    text = f"[{role.upper()}] {term}"
    return f"{text} [DEFINITION] {gloss}" if gloss else text


def _load_encoder_fp32(name_or_path):
    """Load FP32 master weights regardless of dtype recorded by a HF checkpoint."""
    from transformers import AutoModel

    try:
        return AutoModel.from_pretrained(name_or_path, dtype=torch.float32)
    except TypeError:  # transformers<4.56 compatibility
        return AutoModel.from_pretrained(name_or_path, torch_dtype=torch.float32)


class RelationCrossEncoder(nn.Module):
    """Full cross-encoder with a separate parent-independent ROOT head."""

    def __init__(self, cfg: RelationScorerConfig, encoder: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        # Optimizer state must update FP32 master weights. Computation precision is
        # controlled separately by autocast; mixing FP16 encoder parameters with
        # FP32 heads caused both NaNs and checkpoint inference dtype mismatches.
        self.encoder = (encoder if encoder is not None else _load_encoder_fp32(cfg.encoder_name)).float()
        hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.relation_labels = labels_for_objective(cfg.relation_objective)
        self.relation_head = nn.Linear(hidden_size, len(self.relation_labels))
        self.root_head = nn.Linear(hidden_size, 1)

    @staticmethod
    def _pool(output: Any) -> torch.Tensor:
        pooler = getattr(output, "pooler_output", None)
        if pooler is not None:
            return pooler
        return output.last_hidden_state[:, 0]

    def encode(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.dropout(self._pool(self.encoder(**inputs)))

    def forward(
        self,
        pair_inputs: dict[str, torch.Tensor],
        child_inputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        edge_repr = self.encode(pair_inputs)
        relation_logits = self.relation_head(edge_repr)
        output = {"edge_repr": edge_repr, "relation_logits": relation_logits}
        if child_inputs is not None:
            child_repr = self.encode(child_inputs)
            output["child_repr"] = child_repr
            output["root_logit"] = self.root_head(child_repr).squeeze(-1)
        return output

    def edge_potential(self, relation_logits: torch.Tensor) -> torch.Tensor:
        """Convert five logits into a calibrated direct-edge potential."""
        logp = relation_logits.float().log_softmax(dim=-1)
        cfg = self.cfg
        if cfg.relation_objective == "binary":
            return logp[..., self.relation_labels.index("direct")]
        return (
            logp[..., RELATION_TO_ID["direct"]]
            - cfg.ancestor_penalty * logp[..., RELATION_TO_ID["ancestor"]].exp()
            - cfg.reverse_penalty * logp[..., RELATION_TO_ID["reverse"]].exp()
            - cfg.sibling_penalty * logp[..., RELATION_TO_ID["sibling"]].exp()
            - cfg.unrelated_penalty * logp[..., RELATION_TO_ID["unrelated"]].exp()
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(output / "encoder")
        torch.save(
            {
                "config": asdict(self.cfg),
                "relation_head": self.relation_head.state_dict(),
                "root_head": self.root_head.state_dict(),
            },
            output / "relation_heads.pt",
        )

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "RelationCrossEncoder":
        directory = Path(directory)
        payload = torch.load(directory / "relation_heads.pt", map_location="cpu")
        model = cls(
            RelationScorerConfig(**payload["config"]),
            _load_encoder_fp32(directory / "encoder"),
        )
        model.relation_head.load_state_dict(payload["relation_head"])
        model.root_head.load_state_dict(payload["root_head"])
        return model.float()


def _segment_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    result = values.new_zeros((size, values.shape[-1]))
    counts = values.new_zeros((size, 1))
    result.index_add_(0, index, values)
    counts.index_add_(0, index, values.new_ones((values.shape[0], 1)))
    return result / counts.clamp_min(1.0)


class CandidateGraphContextualizer(nn.Module):
    """Two-pass edge model over shared-child and shared-parent neighborhoods."""

    def __init__(
        self,
        edge_dim: int,
        scalar_dim: int = 6,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.edge_projection = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.context = nn.Sequential(
            nn.Linear(3 * hidden_dim + scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        edge_repr: torch.Tensor,
        child_index: torch.Tensor,
        parent_index: torch.Tensor,
        scalar_features: torch.Tensor,
        n_nodes: int | None = None,
    ) -> torch.Tensor:
        if edge_repr.shape[0] == 0:
            return edge_repr.new_zeros((0,))
        size = n_nodes or int(torch.maximum(child_index.max(), parent_index.max()).item()) + 1
        base = self.edge_projection(edge_repr)
        child_context = _segment_mean(base, child_index, size)[child_index]
        parent_context = _segment_mean(base, parent_index, size)[parent_index]
        combined = torch.cat((base, child_context, parent_context, scalar_features), dim=-1)
        return self.context(combined).squeeze(-1)


class RelationStructureModel(nn.Module):
    """Pair scorer + learned graph context producing ROOT-relative potentials."""

    def __init__(self, scorer: RelationCrossEncoder, contextualizer: CandidateGraphContextualizer):
        super().__init__()
        self.scorer = scorer
        self.contextualizer = contextualizer

    def forward(
        self,
        pair_inputs: dict[str, torch.Tensor],
        child_inputs: dict[str, torch.Tensor],
        child_index: torch.Tensor,
        parent_index: torch.Tensor,
        scalar_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local = self.scorer(pair_inputs, child_inputs)
        local_potential = self.scorer.edge_potential(local["relation_logits"])
        context_delta = self.contextualizer(
            local["edge_repr"], child_index, parent_index, scalar_features
        )
        root_logit = local["root_logit"]
        edge_potential = local_potential + context_delta - root_logit[child_index]
        return {
            **local,
            "local_potential": local_potential,
            "context_delta": context_delta,
            "edge_potential": edge_potential,
        }
