"""Reusable frozen-retriever artifacts shared across all training seeds."""

from __future__ import annotations

from pathlib import Path

import torch


SCHEMA_VERSION = 1


def make_record(key: str, nodes: list[str], candidates: list[list[tuple[int, float]]]) -> dict:
    return {"key": key, "nodes": list(nodes),
            "candidates": [[[int(parent), float(score)] for parent, score in row] for row in candidates]}


def save_retrieval_cache(path: Path, *, scope: str, model: str, top_k: int, records: list[dict],
                         metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "scope": scope, "model": model,
                "top_k": int(top_k), "records": records,
                "metadata": metadata or {}}, path)


def load_retrieval_cache(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported retrieval-cache schema in {path}")
    payload["by_key"] = {record["key"]: record for record in payload["records"]}
    return payload


def candidates_for(payload: dict, key: str, nodes: list[str], top_k: int) -> list[list[tuple[int, float]]]:
    if payload["top_k"] < top_k:
        raise ValueError(f"Cache top_k={payload['top_k']} is smaller than requested top_k={top_k}")
    record = payload["by_key"].get(key)
    if record is None:
        raise KeyError(f"Retrieval cache has no record {key!r}")
    if record["nodes"] != list(nodes):
        raise ValueError(f"Node order mismatch for retrieval record {key!r}")
    return [[(int(parent), float(score)) for parent, score in row[:top_k]] for row in record["candidates"]]
