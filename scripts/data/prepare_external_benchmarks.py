#!/usr/bin/env python3
"""Normalize TExEval-2, Schema.org, and MAG transfer benchmarks to JSONL."""

from __future__ import annotations

import argparse
import io
import json
import pickle
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "raw"
OUT = REPO / "data" / "processed" / "taxobranch" / "benchmarks"
SPACE_RE = re.compile(r"[\s_\-]+")


def normalize(term: str) -> str:
    return SPACE_RE.sub(" ", term.strip().casefold()).strip()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare_texeval(raw_root: Path, out_root: Path) -> tuple[list[dict], set[str]]:
    root = raw_root / "texeval2" / "TExEval-2_testdata_1.2"
    terms_dir = root / "gs_terms" / "EN"
    taxo_dir = root / "gs_taxo" / "EN"
    rows: list[dict] = []
    vocabulary: set[str] = set()
    for terms_path in sorted(terms_dir.glob("*.terms")):
        domain = terms_path.stem.removesuffix("_en")
        for line in terms_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if parts:
                vocabulary.add(normalize(parts[-1]))
        taxo_path = taxo_dir / f"{terms_path.stem}.taxo"
        for line in taxo_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            child, parent = parts[1].strip(), parts[2].strip()
            vocabulary.update((normalize(child), normalize(parent)))
            rows.append({
                "benchmark": "texeval2",
                "domain": domain,
                "child": child,
                "parent": parent,
                "child_norm": normalize(child),
                "parent_norm": normalize(parent),
                "split": "test",
            })
    write_jsonl(out_root / "texeval2.jsonl", rows)
    (out_root / "texeval2_vocab.txt").write_text(
        "\n".join(sorted(vocabulary)) + "\n", encoding="utf-8"
    )
    return rows, vocabulary


def _values(value: object) -> list[object]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _identifier(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        item = value.get("@id")
        return item if isinstance(item, str) else None
    return None


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        candidate = value.get("@value")
        return candidate if isinstance(candidate, str) else ""
    if isinstance(value, list):
        return next((_text(item) for item in value if _text(item)), "")
    return ""


def _short(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def prepare_schemaorg(raw_root: Path, out_root: Path) -> tuple[list[dict], set[str]]:
    source = raw_root / "schemaorg" / "schemaorg-current-https.jsonld"
    payload = json.loads(source.read_text(encoding="utf-8"))
    graph = payload.get("@graph", [])
    labels: dict[str, str] = {}
    comments: dict[str, str] = {}
    class_ids: set[str] = set()
    for item in graph:
        if not isinstance(item, dict) or not isinstance(item.get("@id"), str):
            continue
        node_id = item["@id"]
        types = {_identifier(value) for value in _values(item.get("@type"))}
        if any(value and value.endswith("Class") for value in types):
            class_ids.add(node_id)
        labels[node_id] = _text(item.get("rdfs:label")) or _short(node_id)
        comments[node_id] = _text(item.get("rdfs:comment"))
    rows: list[dict] = []
    vocabulary: set[str] = set()
    for item in graph:
        if not isinstance(item, dict) or item.get("@id") not in class_ids:
            continue
        child_id = item["@id"]
        child = labels[child_id]
        vocabulary.add(normalize(child))
        for parent_value in _values(item.get("rdfs:subClassOf")):
            parent_id = _identifier(parent_value)
            if not parent_id or parent_id not in class_ids:
                continue
            parent = labels[parent_id]
            vocabulary.add(normalize(parent))
            rows.append({
                "benchmark": "schemaorg",
                "version": "30.0",
                "child": child,
                "parent": parent,
                "child_id": child_id,
                "parent_id": parent_id,
                "child_gloss": comments.get(child_id, ""),
                "parent_gloss": comments.get(parent_id, ""),
                "child_norm": normalize(child),
                "parent_norm": normalize(parent),
                "split": "validation",
            })
    write_jsonl(out_root / "schemaorg30.jsonl", rows)
    (out_root / "schemaorg30_vocab.txt").write_text(
        "\n".join(sorted(vocabulary)) + "\n", encoding="utf-8"
    )
    return rows, vocabulary


class RestrictedUnpickler(pickle.Unpickler):
    """The benchmark pickles contain only builtin containers and strings."""

    def find_class(self, module: str, name: str):  # pragma: no cover - defensive path
        raise pickle.UnpicklingError(f"global {module}.{name} is forbidden")


def restricted_load(path: Path):
    return RestrictedUnpickler(io.BytesIO(path.read_bytes())).load()


def prepare_mag(raw_root: Path, out_root: Path) -> list[dict]:
    base = raw_root / "taxollama" / "TaxonomyEnrichment" / "data"
    all_rows: list[dict] = []
    for upstream_name, name in (("MAG_CS", "mag_cs"), ("psychology", "mag_psy")):
        task = base / upstream_name
        rows: list[dict] = []
        for line in (task / "train.edgelist").read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            parent, child = parts[:2]  # upstream TaxoExpan orientation
            rows.append({
                "benchmark": name, "child": child, "parent": parent,
                "child_norm": normalize(child), "parent_norm": normalize(parent),
                "child_gloss": "", "split": "seed_taxonomy",
            })
        for record in restricted_load(task / "test_hypernyms_def.pickle"):
            parents = record.get("parents", [])
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                child = str(record["children"])
                rows.append({
                    "benchmark": name, "child": child, "parent": str(parent),
                    "child_norm": normalize(child), "parent_norm": normalize(str(parent)),
                    "child_gloss": str(record.get("child_def", "")), "split": "test",
                })
        write_jsonl(out_root / f"{name}.jsonl", rows)
        all_rows.extend(rows)
    return all_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW)
    parser.add_argument("--out-root", type=Path, default=OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texeval, texeval_vocab = prepare_texeval(args.raw_root, args.out_root)
    schema, schema_vocab = prepare_schemaorg(args.raw_root, args.out_root)
    mag = prepare_mag(args.raw_root, args.out_root)
    stats = {
        "texeval_edges": len(texeval),
        "texeval_vocab": len(texeval_vocab),
        "schemaorg_edges": len(schema),
        "schemaorg_vocab": len(schema_vocab),
        "mag_rows": dict(Counter(row["benchmark"] for row in mag)),
    }
    (args.out_root / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
