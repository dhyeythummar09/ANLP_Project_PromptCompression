"""Build one small, credential-free Kaggle input and its matching Run All notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from final_eval.v3.artifacts import ROOT, read_json
from final_eval.v3.kaggle_bootstrap import MANIFEST, BUNDLE, ARCHIVE, sha

REPO = "Pranav-Sriv/reasoning-critical-span-scorer-v2"
REVISION = "70cc329f10ef6415e152c0bcdcef8894f8dc0d69"


def notebook(manifest_sha, bootstrap_sha):
    install = f'''# First run: leave this False. For recovery, attach the previous notebook's FULL
# saved output and set True. Keep the SAME input ZIP/config and API ledger.
RESUME_FROM_SAVED_OUTPUT = False

from pathlib import Path
import hashlib, json, zipfile

expected = "{manifest_sha}"
source = None
for candidate in sorted(Path("/kaggle/input").rglob("{MANIFEST}")):
    if hashlib.sha256(candidate.read_bytes()).hexdigest() == expected:
        source = candidate.parent
        script = (source / "final_eval/v3/kaggle_bootstrap.py").read_bytes()
        break
if source is None:
    for candidate in sorted(Path("/kaggle/input").rglob("{ARCHIVE}")):
        with zipfile.ZipFile(candidate) as archive:
            if "{MANIFEST}" not in archive.namelist():
                continue
            raw = archive.read("{MANIFEST}")
            if hashlib.sha256(raw).hexdigest() == expected:
                source = candidate
                script = archive.read("final_eval/v3/kaggle_bootstrap.py")
                break
if source is None:
    raise FileNotFoundError("Add the matching symbiote_v3_kaggle.zip as a Kaggle input dataset.")
if hashlib.sha256(script).hexdigest() != "{bootstrap_sha}":
    raise ValueError("Transfer bootstrap checksum mismatch. Use the matching notebook and ZIP.")
namespace = {{"__name__": "verified_v3_bootstrap"}}
exec(compile(script, "verified_v3_bootstrap", "exec"), namespace)
namespace["install"](source, expected)
previous_runs = [p for p in Path("/kaggle/input").rglob("{MANIFEST}")
                 if hashlib.sha256(p.read_bytes()).hexdigest() == expected
                 and (p.parent / "outputs/v3/student_01/resolved_config.json").is_file()]
if previous_runs and not RESUME_FROM_SAVED_OUTPUT:
    raise RuntimeError("A previous full run is attached. Set RESUME_FROM_SAVED_OUTPUT=True to retain its checkpoint/cache/ledger.")
if RESUME_FROM_SAVED_OUTPUT:
    namespace["restore_saved_output"]("/kaggle/input", expected)
'''
    run = '''# No manual decisions between development and test. Default total reader cap: INR500.
# Leave RETRY_FAILED False on the first run. Change only after inspecting an API failure.
RETRY_FAILED = False

import os, subprocess, sys
from pathlib import Path
from kaggle_secrets import UserSecretsClient

try:
    key = UserSecretsClient().get_secret("JARVISLABS_API_KEY")
except Exception:
    raise RuntimeError("Add and ENABLE the JARVISLABS_API_KEY secret under Add-ons > Secrets.") from None
if not key or not key.strip():
    raise RuntimeError("JARVISLABS_API_KEY is empty.")
env = os.environ.copy()
env["HF_HUB_DISABLE_XET"] = "1"  # Avoid Xet-token 404s for legacy model aliases.
env["JARVISLABS_API_KEY"] = key.strip()
del key
root = Path("/kaggle/working/symbiote_v3")
command = [sys.executable, "-u", "-m", "final_eval.v3.kaggle_runner", "run"]
if RETRY_FAILED:
    command.append("--retry-failed")
result = subprocess.run(command, cwd=root, env=env)
del env
if result.returncode:
    raise RuntimeError("Run stopped safely. Keep the notebook output; return v3_status.json and symbiote_v3_results.zip. Do not restart with an empty ledger.")
'''
    review = '''from pathlib import Path
import json
from IPython.display import display, FileLink

working = Path("/kaggle/working")
status = json.loads((working / "v3_status.json").read_text())
print("EXECUTION COMPLETED:", status["success"])
summary = working / "symbiote_v3/outputs/v3/student_01/test/report/summary.json"
if summary.exists():
    report = json.loads(summary.read_text())
    print("Accuracy conclusion:", report["primary"].get("verdict", "inspect report"))
    print(json.dumps(report["primary"].get("result", {}), indent=2))
print("Download the results ZIP and v3_status.json; keep this saved output version.")
display(FileLink(str(working / "symbiote_v3_results.zip")))
display(FileLink(str(working / "v3_status.json")))
'''
    intro = """# Symbiote v3: matched dense students with span criticality

Upload the matching `symbiote_v3_kaggle.zip` as a **private** Kaggle dataset and attach it.
Enable Internet and GPU T4 x2 (uses one GPU). Add and enable the
`JARVISLABS_API_KEY` secret. The key is never put in code or the dataset.

Use **Save Version > Save & Run All** for background execution. Do not also run
the interactive copy: those are separate jobs/ledgers. The run trains two arms,
compares v2/LLMLingua-2, freezes a shared selector using development only, evaluates
the held-out test, measures runtime, and exports results. Reader allowance: INR500.
No dataset recollection. An accuracy improvement is an experimental question.

On failure, preserve the saved output and return `v3_status.json` plus
`symbiote_v3_results.zip`. These are written by the runner even if the final cell
never executes. The full output tree retains optimizer checkpoints for resume.
"""
    cells = [{"id": "v3-intro", "cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}]
    for cell_id, code in zip(("v3-install", "v3-run", "v3-review"), (install, run, review)):
        compile(code, "kaggle_cell", "exec")
        cells.append({"id": cell_id, "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                      "source": code.splitlines(keepends=True)})
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}


def build(root=ROOT, output=None):
    root = Path(root).resolve()
    output = Path(output or root / "outputs/handoffs/v3_20261030")
    config = read_json(root / "final_eval/v3/configs/student.json")
    if config["output_dir"] != "outputs/v3/student_01" or config["project_root"] != ".":
        raise ValueError("Kaggle transfer expects project_root='.' and outputs/v3/student_01")
    model = root / config["v2_model_dir"]
    weights = {f"fold{f}/scorer.pt": {"sha256": sha(model / f"fold{f}/scorer.pt"),
                                      "bytes": (model / f"fold{f}/scorer.pt").stat().st_size} for f in range(5)}
    verification = read_json(root / "outputs/hf_release/remote_release_verification.json")
    if not verification.get("complete") or verification["repo_id"] != REPO or verification["revision"] != REVISION:
        raise ValueError("Need the verified published five-fold v2 release")
    # Check against the exact already-published release, not merely another local checkpoint.
    release = root / "outputs/hf_release/span_scorer_v2_20261021"
    for name, info in weights.items():
        if sha(release / name) != info["sha256"]:
            raise ValueError(f"Local reference differs from the published release: {name}")
    files = sorted((root / "final_eval").rglob("*.py"))
    files += sorted((root / "final_eval").glob("requirements*.txt"))
    files += sorted((root / "final_eval/v3").glob("*.txt"))
    files += sorted((root / "final_eval/v3/configs").glob("*.json"))
    files += [root / "README.md", root / "requirements.txt"]
    files += [root / config[k] for k in ("problems", "span_labels", "teacher_cache")]
    files += [root / name for name in config["exposure_files"]]
    files += [p for p in (root / "results").rglob("exposure.json")
              if read_json(p).get("problems_sha256") == sha(root / config["problems"])]
    files += [model / "split.json", model / "training_metadata.json"]
    files += sorted((model / "tokenizer").glob("*.json"))
    for f in range(5):
        files += [model / f"fold{f}/{name}" for name in ("config.json", "category_prior.json")]
    files = sorted(set(files))
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Missing source/data/model metadata required by the Kaggle transfer")
    names = [p.relative_to(root).as_posix() for p in files]
    if any("__pycache__" in p.parts or p.suffix in {".env", ".pt"} for p in files):
        raise ValueError("Unexpected credentials/weights/cache in the small upload payload")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    manifest = {"bundle": BUNDLE, "git_commit": commit,
        "files": {name: {"sha256": sha(path), "bytes": path.stat().st_size} for name, path in zip(names, files)},
        "v2_weights": {"repo_id": REPO, "revision": REVISION, "files": weights},
        "scope": "Existing merged data, exact code and metadata. No credentials or prior API raw responses. Five published v2 weights are downloaded and hash-checked on Kaggle."}
    raw = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    target = output / ARCHIVE
    if target.exists():
        with zipfile.ZipFile(target) as old:
            if old.read(MANIFEST) != raw:
                raise FileExistsError("This transfer version already exists with different inputs. Use --output for a new version; preserve the old ZIP/notebook pair.")
    else:
        temp = target.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for name, path in zip(names, files):
                archive.write(path, name)
            archive.writestr(MANIFEST, raw)
        temp.replace(target)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError("Transfer CRC failed")
        for name, info in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != info["sha256"]:
                raise ValueError(f"Transfer verification failed: {name}")
    book = notebook(hashlib.sha256(raw).hexdigest(), sha(root / "final_eval/v3/kaggle_bootstrap.py"))
    encoded = json.dumps(book, indent=1, ensure_ascii=False) + "\n"
    (output / "Symbiote_V3_Run_All.ipynb").write_text(encoded, encoding="utf-8")
    (root / "notebooks").mkdir(exist_ok=True)
    (root / "notebooks/05_v3_students_kaggle.ipynb").write_text(encoded, encoding="utf-8")
    record = {"archive": str(target), "bytes": target.stat().st_size, "sha256": sha(target),
              "manifest_sha256": hashlib.sha256(raw).hexdigest(), "files": len(files),
              "notebook": str(output / "Symbiote_V3_Run_All.ipynb"), "git_commit": commit,
              "new_api_calls": 0, "model_training": False}
    (output / "build_verification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return target, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    build(output=args.output)


if __name__ == "__main__":
    main()
