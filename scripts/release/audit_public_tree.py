#!/usr/bin/env python3
"""Fail-fast audit for a TaxoBranch public-release tree."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


MAX_FILE_BYTES = 20 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".bin", ".ckpt", ".key", ".pem", ".pt", ".pth", ".safetensors", ".aux", ".bbl", ".blg", ".log", ".out", ".pyc"}
FORBIDDEN_NAMES = {".env", "credentials.json", "secrets.json"}
PRIVATE_PATTERNS = {
    "local absolute path": re.compile(r"/(?:home|workspace(?:-[^/]+)?|mnt|scratch)/[^/\s]+(?:/|\\b)"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
}
SKIP_DIRS = {".git"}
FORBIDDEN_ARTIFACT_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}


@dataclass(frozen=True)
class AuditIssue:
    path: str
    reason: str


def audit_tree(root: Path) -> dict[str, object]:
    root = root.resolve()
    issues: list[AuditIssue] = []
    files = 0
    total_bytes = 0

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if any(
            part in FORBIDDEN_ARTIFACT_DIRS or part.endswith(".egg-info")
            for part in relative.parts
        ):
            issues.append(AuditIssue(str(relative), "generated build or cache artifact"))
            continue
        if path.is_symlink():
            try:
                path.resolve().relative_to(root)
            except (OSError, ValueError):
                issues.append(AuditIssue(str(relative), "symlink resolves outside release tree"))
            continue
        if not path.is_file():
            continue

        files += 1
        size = path.stat().st_size
        total_bytes += size
        lower_name = path.name.lower()
        if lower_name in FORBIDDEN_NAMES or lower_name.startswith(".env."):
            issues.append(AuditIssue(str(relative), "secret-bearing filename"))
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            issues.append(AuditIssue(str(relative), "model weight or private-key file"))
        if size > MAX_FILE_BYTES:
            issues.append(AuditIssue(str(relative), f"file exceeds {MAX_FILE_BYTES // 1024 // 1024} MiB"))
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(content):
                issues.append(AuditIssue(str(relative), label))

    return {
        "root": str(root),
        "files": files,
        "bytes": total_bytes,
        "issues": [asdict(issue) for issue in issues],
        "ok": not issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    report = audit_tree(args.root)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
