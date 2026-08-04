"""Minimal, version-pinned reader for Princeton WordNet 3.0 database files.

This avoids silently changing the graph when a machine has a different NLTK
snapshot.  Only noun/verb synsets and hypernym pointers are needed by the
relation-supervision builder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path


POS_FILE = {"n": "data.noun", "v": "data.verb"}
HYPERNYM_POINTERS = {"@", "@i"}


@dataclass(frozen=True)
class SynsetNode:
    synset_id: str
    offset: str
    pos: str
    lex_filenum: int
    lexname: str
    lemmas: tuple[str, ...]
    gloss: str

    @property
    def surface(self) -> str:
        return self.lemmas[0].replace("_", " ")


@dataclass
class WordNetGraph:
    nodes: dict[str, SynsetNode]
    parents: dict[str, set[str]]

    @property
    def edges(self) -> set[tuple[str, str]]:
        return {(child, parent) for child, values in self.parents.items() for parent in values}

    def ancestors(self, child: str, max_distance: int = 4) -> dict[str, int]:
        distances: dict[str, int] = {}
        queue = deque((parent, 1) for parent in self.parents.get(child, ()))
        while queue:
            node, distance = queue.popleft()
            if distance > max_distance or distance >= distances.get(node, max_distance + 1):
                continue
            distances[node] = distance
            if distance < max_distance:
                queue.extend((parent, distance + 1) for parent in self.parents.get(node, ()))
        return distances

    def induced(self, keep: set[str]) -> "WordNetGraph":
        return WordNetGraph(
            nodes={node: value for node, value in self.nodes.items() if node in keep},
            parents={
                child: {parent for parent in values if parent in keep}
                for child, values in self.parents.items()
                if child in keep
            },
        )


def _lexnames(dict_dir: Path) -> dict[int, str]:
    path = dict_dir / "lexnames"
    if not path.exists():
        # The DB-only archive omits lexnames on some mirrors.  WordNet's
        # lexicographer file numbers are stable, so preserve a deterministic
        # domain label even without the friendly name.
        return {i: f"lexfile_{i:02d}" for i in range(45)}
    result: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            result[int(parts[0])] = parts[1]
    return result


def load_wordnet_graph(dict_dir: Path) -> WordNetGraph:
    names = _lexnames(dict_dir)
    nodes: dict[str, SynsetNode] = {}
    raw_pointers: list[tuple[str, str, str, str]] = []

    for expected_pos, filename in POS_FILE.items():
        path = dict_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing WordNet database file: {path}")
        with path.open(encoding="utf-8") as stream:
            for raw in stream:
                if not raw or raw[0].isspace() or raw.startswith("  "):
                    continue
                body, _, gloss = raw.rstrip("\n").partition(" | ")
                fields = body.split()
                if len(fields) < 5 or not fields[0].isdigit():
                    continue
                offset, lex_field, synset_type = fields[:3]
                pos = "a" if synset_type == "s" else synset_type
                if pos != expected_pos:
                    continue
                word_count = int(fields[3], 16)
                cursor = 4
                lemmas = tuple(fields[cursor + 2 * i] for i in range(word_count))
                cursor += 2 * word_count
                pointer_count = int(fields[cursor])
                cursor += 1
                synset_id = f"{pos}:{offset}"
                lex_filenum = int(lex_field)
                nodes[synset_id] = SynsetNode(
                    synset_id=synset_id,
                    offset=offset,
                    pos=pos,
                    lex_filenum=lex_filenum,
                    lexname=names.get(lex_filenum, f"{pos}.lexfile_{lex_filenum:02d}"),
                    lemmas=lemmas,
                    gloss=gloss,
                )
                for _ in range(pointer_count):
                    symbol, target_offset, target_pos, source_target = fields[cursor : cursor + 4]
                    cursor += 4
                    raw_pointers.append((synset_id, symbol, target_offset, target_pos))

    parents: dict[str, set[str]] = {node: set() for node in nodes}
    for child, symbol, target_offset, target_pos in raw_pointers:
        if symbol not in HYPERNYM_POINTERS:
            continue
        target_pos = "a" if target_pos == "s" else target_pos
        parent = f"{target_pos}:{target_offset}"
        if parent in nodes:
            parents[child].add(parent)
    return WordNetGraph(nodes=nodes, parents=parents)
