"""Verified Kaggle transfer installation. Uses only the Python standard library."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

MANIFEST = "V3_KAGGLE_MANIFEST.json"
BUNDLE = "symbiote-v3-kaggle-v1"
ARCHIVE = "symbiote_v3_kaggle.zip"
KAGGLE_ROOT = "/kaggle/working/symbiote_v3"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def relative(name):
    value = PurePosixPath(name)
    if not value.parts or value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe transfer path: {name}")
    return Path(*value.parts)


def find_payload(inputs, expected_manifest):
    """Accept Kaggle's expanded upload or the unchanged ZIP; ignore other inputs."""
    inputs = Path(inputs)
    candidates = sorted(inputs.rglob(MANIFEST)) + sorted(inputs.rglob(ARCHIVE))
    for candidate in candidates:
        if candidate.name == MANIFEST:
            raw = candidate.read_bytes()
            source = candidate.parent
        else:
            with zipfile.ZipFile(candidate) as archive:
                if MANIFEST not in archive.namelist():
                    continue
                raw = archive.read(MANIFEST)
            source = candidate
        if hashlib.sha256(raw).hexdigest() == expected_manifest:
            return source
    raise FileNotFoundError("Attach the matching symbiote_v3_kaggle.zip dataset. No matching transfer manifest found.")


def install(source, expected_manifest, root=KAGGLE_ROOT):
    source, root = Path(source), Path(root).resolve()
    archive = zipfile.ZipFile(source) if source.is_file() else None
    try:
        if archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or archive.testzip() is not None:
                raise ValueError("Duplicate entries or damaged transfer ZIP")
        raw = archive.read(MANIFEST) if archive else (source / MANIFEST).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest:
            raise ValueError("Wrong transfer version: import the notebook shipped with this ZIP")
        manifest = json.loads(raw)
        if manifest.get("bundle") != BUNDLE:
            raise ValueError("Wrong transfer bundle")
        if archive and set(names) != set(manifest["files"]) | {MANIFEST}:
            raise ValueError("Unexpected or missing transfer members")
        if (root / MANIFEST).exists() and sha(root / MANIFEST) != expected_manifest:
            raise ValueError("Existing working tree belongs to another transfer. Preserve it; use a fresh session.")
        with tempfile.TemporaryDirectory(prefix="symbiote_v3_verified_") as temp:
            staged = Path(temp)
            for name, info in manifest["files"].items():
                rel = relative(name)
                path = staged / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                with (archive.open(name) if archive else (source / rel).open("rb")) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
                    raise ValueError(f"Transfer checksum mismatch: {name}")
                destination = root / rel
                if destination.exists() and sha(destination) != info["sha256"]:
                    raise ValueError(f"Existing file differs; preserved without replacement: {destination}")
            # Every input is checked BEFORE any existing source/data is touched.
            for name in manifest["files"]:
                dest = root / relative(name)
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(staged / relative(name), dest)
            root.mkdir(parents=True, exist_ok=True)
            (root / MANIFEST).write_bytes(raw)
        print(f"Installed and verified {len(manifest['files'])} files at {root}", flush=True)
        return root
    finally:
        if archive:
            archive.close()


def restore_saved_output(inputs, expected_manifest, root=KAGGLE_ROOT):
    """Explicit opt-in resume from a previous saved notebook's FULL output tree."""
    root = Path(root).resolve()
    candidates = []
    for marker in Path(inputs).rglob(MANIFEST):
        if sha(marker) == expected_manifest and (marker.parent / "outputs/v3/student_01/resolved_config.json").is_file():
            candidates.append(marker.parent)
    if len(candidates) != 1:
        raise ValueError("Resume needs exactly one previous saved notebook output with the full symbiote_v3 tree. A results ZIP omits optimizer state and is insufficient.")
    old = candidates[0] / "outputs/v3"
    target = root / "outputs/v3"
    for src in sorted(old.rglob("*")):
        if not src.is_file() or src.suffix in {".lock", ".tmp"}:
            continue
        dest = target / src.relative_to(old)
        if dest.exists() and sha(dest) != sha(src):
            raise ValueError(f"Resume would replace existing work: {dest}")
    for src in sorted(old.rglob("*")):
        if src.is_file() and src.suffix not in {".lock", ".tmp"}:
            dest = target / src.relative_to(old)
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
    print("Restored full run, including checkpoints, exposure markers and the existing API ledger.", flush=True)


def fetch_weights(root, downloader=None):
    """Download ONLY the five published checkpoints, pinned and locally verified."""
    root = Path(root)
    spec = json.loads((root / MANIFEST).read_text(encoding="utf-8"))["v2_weights"]
    if downloader is None:
        from huggingface_hub import hf_hub_download
        downloader = hf_hub_download
    model = root / "models/span_scorer_v2_20261021"
    for name, info in spec["files"].items():
        dest = model / relative(name)
        if not dest.exists():
            downloader(repo_id=spec["repo_id"], revision=spec["revision"], filename=name,
                       local_dir=str(model), token=False)
        if not dest.is_file() or dest.stat().st_size != info["bytes"] or sha(dest) != info["sha256"]:
            raise ValueError(f"Published checkpoint differs from the verified v2 artifact: {name}")
    print("Five pinned v2 checkpoints verified; no retraining or replacement.", flush=True)
