"""Role-conditioned dense retrieval for candidate parents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.models.relation_aware import render_role_text


@dataclass
class RetrievalItem:
    term: str
    gloss: str = ""


class RoleConditionedRetriever:
    """Shared encoder with explicit child/parent prompts and cosine search."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "auto"):
        from transformers import AutoModel, AutoTokenizer

        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def encode(self, items: list[RetrievalItem], role: str, batch_size: int = 64) -> torch.Tensor:
        vectors = []
        for start in range(0, len(items), batch_size):
            texts = [render_role_text(item.term, item.gloss, role) for item in items[start : start + batch_size]]
            batch = self.tokenizer(
                texts, padding=True, truncation=True, max_length=192, return_tensors="pt"
            ).to(self.device)
            output = self.model(**batch).last_hidden_state
            last = batch["attention_mask"].sum(dim=1) - 1
            pooled = output[torch.arange(output.shape[0], device=self.device), last]
            vectors.append(F.normalize(pooled.float(), dim=-1).cpu())
        return torch.cat(vectors) if vectors else torch.empty((0, 0))

    def retrieve(
        self,
        children: list[RetrievalItem],
        parents: list[RetrievalItem],
        top_k: int = 50,
        exclude_identity: bool = True,
    ) -> list[list[tuple[int, float]]]:
        child_vectors = self.encode(children, "child")
        parent_vectors = self.encode(parents, "parent")
        scores = child_vectors @ parent_vectors.T
        if exclude_identity:
            parent_terms: dict[str, list[int]] = {}
            for index, item in enumerate(parents):
                parent_terms.setdefault(item.term.casefold(), []).append(index)
            for child_index, item in enumerate(children):
                for parent_index in parent_terms.get(item.term.casefold(), []):
                    scores[child_index, parent_index] = -torch.inf
        values, indices = scores.topk(min(top_k, len(parents)), dim=1)
        return [[(int(i), float(s)) for i, s in zip(ix, val)] for ix, val in zip(indices, values)]


def _lexical_tokens(item: RetrievalItem) -> set[str]:
    import re

    return {
        token
        for token in re.findall(r"[\w]+", f"{item.term} {item.gloss}".casefold())
        if len(token) > 2
    }


def hybrid_retrieve(
    retriever: RoleConditionedRetriever,
    children: list[RetrievalItem],
    parents: list[RetrievalItem],
    top_k: int = 200,
    rrf_constant: int = 60,
) -> list[list[tuple[int, float]]]:
    """Fuse dense retrieval with sparse lexical discovery using deterministic RRF."""
    dense = retriever.retrieve(children, parents, top_k, exclude_identity=True)
    parent_tokens = [_lexical_tokens(item) for item in parents]
    inverted: dict[str, set[int]] = {}
    for parent_index, tokens in enumerate(parent_tokens):
        for token in tokens:
            inverted.setdefault(token, set()).add(parent_index)
    output = []
    for child_index, (child, dense_row) in enumerate(zip(children, dense)):
        child_tokens = _lexical_tokens(child)
        lexical_pool: set[int] = set()
        for token in child_tokens:
            lexical_pool.update(inverted.get(token, ()))
        lexical_pool.discard(child_index)
        lexical = sorted(
            lexical_pool,
            key=lambda parent_index: (
                -len(child_tokens & parent_tokens[parent_index]) / max(1, len(child_tokens | parent_tokens[parent_index])),
                parent_index,
            ),
        )
        fused: dict[int, float] = {}
        for rank, (parent_index, _) in enumerate(dense_row, 1):
            fused[parent_index] = fused.get(parent_index, 0.0) + 1.0 / (rrf_constant + rank)
        for rank, parent_index in enumerate(lexical, 1):
            fused[parent_index] = fused.get(parent_index, 0.0) + 1.0 / (rrf_constant + rank)
        ranked = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        output.append([(parent_index, score) for parent_index, score in ranked])
    return output
