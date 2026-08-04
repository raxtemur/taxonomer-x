"""Label-free parent-conditioned retrieval for a complementary direct parent."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.models.retriever import RetrievalItem, RoleConditionedRetriever


@dataclass(frozen=True)
class ConditionalRetrievalPolicy:
    base_k: int = 20
    primary_beam: int = 3
    conditional_k: int = 10
    output_k: int = 50

    def __post_init__(self) -> None:
        if min(self.base_k, self.primary_beam, self.conditional_k) < 1:
            raise ValueError("retrieval budgets must be positive")
        if self.output_k < self.base_k:
            raise ValueError("output_k must be at least base_k")


def conditional_query(child: RetrievalItem, primary: RetrievalItem) -> RetrievalItem:
    return RetrievalItem(
        term=(
            f"{child.term} [KNOWN_PARENT] {primary.term} "
            "[TASK] alternative direct parent"
        ),
        gloss=(
            f"Child definition: {child.gloss} "
            f"Known-parent definition: {primary.gloss}"
        ).strip(),
    )


def conditional_second_parent_retrieve(
    retriever: RoleConditionedRetriever,
    items: list[RetrievalItem],
    base_candidates: list[list[tuple[int, float]]],
    policy: ConditionalRetrievalPolicy,
) -> list[list[tuple[int, float]]]:
    """Append conditional candidates while preserving the dense prefix exactly."""
    if len(items) != len(base_candidates):
        raise ValueError("items and base candidates must align")
    queries: list[RetrievalItem] = []
    references: list[tuple[int, int]] = []
    for child_index, row in enumerate(base_candidates):
        for primary_index, _ in row[: policy.primary_beam]:
            queries.append(conditional_query(items[child_index], items[int(primary_index)]))
            references.append((child_index, int(primary_index)))

    by_child: list[dict[int, float]] = [dict() for _ in items]
    if queries and items:
        query_vectors = retriever.encode(queries, "second_parent")
        parent_vectors = retriever.encode(items, "parent")
        scores = query_vectors @ parent_vectors.T
        width = min(len(items), policy.conditional_k + 2)
        values, indices = scores.topk(width, dim=1)
        for (child_index, primary_index), candidates, candidate_scores in zip(
            references, indices.tolist(), values.tolist()
        ):
            accepted = 0
            for candidate_index, score in zip(candidates, candidate_scores):
                if candidate_index in {child_index, primary_index}:
                    continue
                previous = by_child[child_index].get(candidate_index)
                if previous is None or score > previous:
                    by_child[child_index][candidate_index] = float(score)
                accepted += 1
                if accepted == policy.conditional_k:
                    break

    output = []
    for child_index, row in enumerate(base_candidates):
        prefix = []
        seen = set()
        for parent_index, score in row[: policy.base_k]:
            parent_index = int(parent_index)
            if parent_index not in seen:
                prefix.append((parent_index, float(score)))
                seen.add(parent_index)
        residual = sorted(
            (
                (parent_index, score)
                for parent_index, score in by_child[child_index].items()
                if parent_index not in seen and parent_index != child_index
            ),
            key=lambda item: (-item[1], item[0]),
        )
        output.append((prefix + residual)[: policy.output_k])
    return output


def prefix_is_preserved(
    base: list[list[tuple[int, float]]],
    augmented: list[list[tuple[int, float]]],
    base_k: int,
) -> bool:
    return all(
        [(p, float(s)) for p, s in old[:base_k]]
        == [(p, float(s)) for p, s in new[: min(base_k, len(old))]]
        for old, new in zip(base, augmented)
    )
