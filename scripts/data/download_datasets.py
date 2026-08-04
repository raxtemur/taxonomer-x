#!/usr/bin/env python3
"""Download, verify, and unpack the TaxoBranch data snapshot.

The raw data directory is intentionally git-ignored.  Reproducibility lives in
``configs/data_sources.json``: immutable version identifiers, upstream URLs and
SHA-256 digests.  Google Drive is used only for the TaxoLLaMA conversion; a
verified local cache can be supplied to avoid Drive failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "configs" / "data_sources.json"
DEFAULT_RAW = REPO / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified(path: Path, expected: str) -> bool:
    return path.is_file() and sha256(path) == expected


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent,
        delete=False,
    ) as tmp:
        temporary = Path(tmp.name)
    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "taxonomer-x-data/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def copy_from_cache(filename: str, cache_dir: Path | None, destination: Path) -> bool:
    if cache_dir is None:
        return False
    matches = list(cache_dir.rglob(Path(filename).name))
    if not matches:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], destination)
    return True


def download_drive_folder(folder_id: str, filename: str, destination: Path) -> None:
    try:
        import gdown  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TaxoLLaMA is a Google Drive folder. Install the optional 'gdown' "
            "package or pass --cache-dir containing data_taxonomy_enrichment.tar.gz."
        ) from exc
    with tempfile.TemporaryDirectory(prefix="taxonomerx-gdrive-") as temp_dir:
        outputs = gdown.download_folder(
            id=folder_id, output=temp_dir, quiet=False, remaining_ok=True
        )
        candidates = [Path(p) for p in (outputs or []) if Path(p).name == filename]
        if not candidates:
            candidates = list(Path(temp_dir).rglob(filename))
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one {filename!r} in Drive folder, found {len(candidates)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name!r}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Links are not accepted in data archives: {member.name!r}")
        bundle.extractall(destination)


def acquire_one(name: str, spec: dict, raw_root: Path, cache_dir: Path | None) -> dict:
    archive = raw_root / spec["archive"]
    expected = spec["sha256"]
    if not verified(archive, expected):
        archive.unlink(missing_ok=True)
        cached = copy_from_cache(spec["archive"], cache_dir, archive)
        if not cached and "url" in spec:
            print(f"[{name}] downloading {spec['url']}", flush=True)
            download(spec["url"], archive)
        elif not cached and "google_drive_folder" in spec:
            print(f"[{name}] downloading Google Drive folder", flush=True)
            download_drive_folder(
                spec["google_drive_folder"], Path(spec["archive"]).name, archive
            )
        if not verified(archive, expected):
            actual = sha256(archive) if archive.exists() else "missing"
            raise RuntimeError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")
    else:
        print(f"[{name}] verified cache: {archive}", flush=True)

    if spec.get("extract_to"):
        safe_extract(archive, raw_root / spec["extract_to"])
    return {
        "name": name,
        "version": spec["version"],
        "path": str(archive.relative_to(REPO)),
        "sha256": expected,
        "bytes": archive.stat().st_size,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--datasets", nargs="*", help="Subset of manifest keys")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    specs = manifest["datasets"]
    selected = args.datasets or list(specs)
    unknown = sorted(set(selected) - set(specs))
    if unknown:
        raise SystemExit(f"Unknown datasets: {', '.join(unknown)}")
    resolved = [acquire_one(name, specs[name], args.raw_root, args.cache_dir) for name in selected]
    receipt = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        "datasets": resolved,
    }
    receipt_path = args.raw_root / "DOWNLOAD_RECEIPT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {receipt_path}")


if __name__ == "__main__":
    main()
