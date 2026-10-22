"""Install a verified Stage C transfer from a Kaggle dataset, using only stdlib."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile

MANIFEST = "MAIN_TRANSFER_MANIFEST.json"
BUNDLE = "symbiote-main-stage-c-v1"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name or not path.parts:
        raise ValueError(f"Unsafe transfer path: {name}")
    return Path(*path.parts)


def install(source, expected_manifest_sha, root="/kaggle/working/symbiote"):
    """Verify every payload before touching the existing working directory."""
    source, root = Path(source), Path(root).resolve()
    archive = zipfile.ZipFile(source) if source.is_file() else None
    if archive and (archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist()))):
        archive.close()
        raise ValueError("Transfer ZIP failed CRC/duplicate-entry check")
    try:
        raw = archive.read(MANIFEST) if archive else (source / MANIFEST).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha:
            raise ValueError("This is a different transfer version; use the notebook cells shipped with this ZIP")
        manifest = json.loads(raw)
        if manifest["bundle"] != BUNDLE:
            raise ValueError("Wrong transfer bundle")
        with tempfile.TemporaryDirectory(prefix="symbiote_main_verified_") as temporary:
            staging = Path(temporary)
            for name, expected in manifest["files"].items():
                relative = safe_relative(name)
                path = staging / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                with (archive.open(name) if archive else (source / relative).open("rb")) as stream, path.open("wb") as output:
                    shutil.copyfileobj(stream, output, 1024 * 1024)
                if sha(path) != expected:
                    raise ValueError(f"Transfer SHA256 mismatch: {name}")
                destination = root / relative
                # Existing data/model metadata and frozen outcomes must be identical.
                if destination.exists() and name.startswith(("data/", "models/", "outputs/")) and sha(destination) != expected:
                    raise ValueError(f"Existing artifact differs; preserved without replacement: {destination}")
            root.mkdir(parents=True, exist_ok=True)
            backup = root / "outputs/final_eval/main_gptoss_v1/transfer_source_backups"
            for name, expected in manifest["files"].items():
                relative = safe_relative(name)
                destination = root / relative
                if destination.exists():
                    observed = sha(destination)
                    if observed == expected:
                        continue
                    saved = backup / observed / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(destination, saved)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging / relative, destination)
                if sha(destination) != expected:
                    raise ValueError(f"Installed file did not match: {name}")
            (root / MANIFEST).write_bytes(raw)
        print(f"Verified and installed {len(manifest['files'])} files at {root}.")
        print("Existing scorer weights and previous results preserved. Next run the SETUP cell.")
        return root
    finally:
        if archive:
            archive.close()
