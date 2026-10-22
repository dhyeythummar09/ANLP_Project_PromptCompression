"""Build the credential-free main transfer and fresh-kernel notebook locally."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from final_eval.utils.artifacts import file_hash, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/handoffs"
NAME = "symbiote_main_stage_c_v1.zip"


def cells(script_sha, manifest_sha):
    install = f'''from pathlib import Path
import hashlib, zipfile

inputs = Path("/kaggle/input")
expanded = sorted(inputs.rglob("kaggle_stage_c_bootstrap.py"))
archives = sorted(inputs.rglob("{NAME}"))
if expanded:
    source = expanded[0].parent.parent
    script = expanded[0].read_bytes()
elif archives:
    source = archives[0]
    with zipfile.ZipFile(source) as archive:
        script = archive.read("final_eval/kaggle_stage_c_bootstrap.py")
else:
    raise FileNotFoundError("Attach the NEW main-stage transfer ZIP as a Kaggle input dataset first.")
assert hashlib.sha256(script).hexdigest() == "{script_sha}", "Wrong transfer version; use the matching cells"
namespace = {{"__name__": "main_transfer_bootstrap"}}
exec(compile(script, "verified_stage_c_bootstrap", "exec"), namespace)
namespace["install"](source, "{manifest_sha}")
'''
    setup = '''from pathlib import Path
import subprocess, sys

root = Path("/kaggle/working/symbiote")
subprocess.run([sys.executable, "-u", "-m", "final_eval.kaggle_main", "setup"], cwd=root, check=True)
'''
    run = '''from pathlib import Path
import subprocess, sys

root = Path("/kaggle/working/symbiote")
subprocess.run([sys.executable, "-u", "-m", "final_eval.kaggle_main", "run"], cwd=root, check=True)
'''
    download = '''from pathlib import Path
import subprocess, sys
from IPython.display import display, FileLink

root = Path("/kaggle/working/symbiote")
# Safe to run alone after an error/interruption; this only exports saved files.
subprocess.run([sys.executable, "-u", "-m", "final_eval.kaggle_main", "export"], cwd=root, check=True)
archives = sorted(Path("/kaggle/working").glob("symbiote_main_review_*.zip"), key=lambda p: p.stat().st_mtime)
print("Download this ZIP and put it in the local project folder:", archives[-1])
display(FileLink(str(archives[-1])))
'''
    return [install, setup, run, download]


def build():
    # V1 is a recorded experiment, not a rolling archive of the checkout.
    # Documentation/packaging changes must not replace its executed source bytes.
    target = OUT / NAME
    if target.exists():
        with zipfile.ZipFile(target) as archive:
            raw = archive.read("MAIN_TRANSFER_MANIFEST.json")
            manifest = json.loads(raw)
            if manifest.get("bundle") != "symbiote-main-stage-c-v1" or archive.testzip() is not None:
                raise ValueError("Invalid frozen main transfer")
            for name, expected in manifest["files"].items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                    raise ValueError(f"Frozen transfer checksum mismatch: {name}")
        blocks = cells(manifest["files"]["final_eval/kaggle_stage_c_bootstrap.py"],
                       hashlib.sha256(raw).hexdigest())
        write_notebook(blocks)
        print(f"Reused verified frozen transfer: {target}; source and model artifacts unchanged.")
        return target, manifest, blocks
    files = []
    package = ROOT / "final_eval"
    for directory in ("compressor", "inference", "scorer", "spans", "utils", "preparation"):
        files.extend(sorted((package / directory).rglob("*.py")))
    files += [package / name for name in ("__init__.py", "settings.py", "batch.py", "handoff.py",
                "kaggle_main.py", "kaggle_stage_c_bootstrap.py")]
    files += sorted(package.glob("requirements*.txt"))
    model = ROOT / "models/span_scorer_v2_20261021"
    files += [model / "split.json", model / "training_metadata.json"]
    files += sorted(model.glob("fold*/*.json")) + sorted((model / "tokenizer").glob("*.json"))
    files += [ROOT / p for p in ("data/processed/team_v2_20261021/consensus_dataset.csv",
        "outputs/final_eval/main_gptoss_v1/cohort.csv",
        "outputs/final_eval/main_gptoss_v1/cohort.csv.cohort.json",
        "outputs/final_eval/main_gptoss_v1/preparation_check.json",
        "outputs/final_eval/target_pilots/gptoss/check.json")]
    files = sorted(set(files))
    if any(not p.is_file() for p in files):
        raise FileNotFoundError("Required prepared artifacts missing")
    checkpoints = {p.relative_to(ROOT).as_posix():{"sha256":file_hash(p), "bytes":p.stat().st_size}
                   for p in sorted(model.glob("fold*/scorer.pt"))}
    if len(checkpoints) != 5:
        raise ValueError("Five trained checkpoints must be backed up locally before transfer")
    gate = json.loads((ROOT / "outputs/final_eval/target_pilots/gptoss/check.json").read_text())
    if not gate["passed"]:
        raise ValueError("Need passed target pilot")
    manifest = {"bundle":"symbiote-main-stage-c-v1", "source_fingerprint":source_fingerprint(),
        "files":{p.relative_to(ROOT).as_posix():file_hash(p) for p in files}, "checkpoints":checkpoints,
        "scope":"Main compression only. Scorer checkpoints reused on Kaggle, not included. No credentials, training or target API calls."}
    raw = (json.dumps(manifest,indent=2) + "\n").encode()
    OUT.mkdir(parents=True,exist_ok=True)
    target = OUT / NAME
    if target.exists():
        with zipfile.ZipFile(target) as existing:
            if existing.read("MAIN_TRANSFER_MANIFEST.json") != raw:
                raise FileExistsError("Transfer version already exists with different inputs; preserve/version it before rebuilding")
    else:
        with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
            for p in files:
                archive.write(p,p.relative_to(ROOT).as_posix())
            archive.writestr("MAIN_TRANSFER_MANIFEST.json",raw)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        assert not any(name.endswith((".pt",".env")) for name in archive.namelist())
        for name, expected in manifest["files"].items():
            h=hashlib.sha256()
            with archive.open(name) as stream:
                for part in iter(lambda:stream.read(1024*1024),b""):
                    h.update(part)
            assert h.hexdigest()==expected,name
    blocks = cells(file_hash(package/"kaggle_stage_c_bootstrap.py"),hashlib.sha256(raw).hexdigest())
    write_notebook(blocks)
    print(f"Transfer ready: {target} ({target.stat().st_size/1024**2:.2f} MiB); {len(files)} verified payload files, no weights/keys.")
    return target,manifest,blocks


def write_notebook(blocks):
    instructions = """# Recorded main compression workflow

This notebook reproduces the completed 21 October Kaggle compression stage.
Read `README.md` in the repository first. Existing final results need no new GPU run.

For an intentional reproduction, attach the preserved
`outputs/handoffs/symbiote_main_stage_c_v1.zip` as a Kaggle input. Restore the five
recorded checkpoints under `/kaggle/working/symbiote/models/span_scorer_v2_20261021/`
before setup; they are not included in the transfer. Enable GPU and Internet.
Run the four cells in order. Setup must pass before compression. The last cell
only exports saved artifacts and can run independently after a failure.

The transfer pins the original source bytes and manifest, preserves outputs,
and verifies checkpoints. It generates 2,240 compressed prompts, with no target
API calls or training. The two hosted-reader runs are documented in README.md
and RESULTS.md; they have already finished.
"""
    notebook = {"nbformat":4,"nbformat_minor":5,"metadata":{
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
        "language_info":{"name":"python"}},"cells":[{
            "cell_type":"markdown","id":"stage-c-context","metadata":{},"source":instructions.splitlines(keepends=True)}]}
    for i,block in enumerate(blocks,1):
        notebook["cells"].append({"cell_type":"code","execution_count":None,"id":f"stage-c-{i}","metadata":{},"outputs":[],"source":block.splitlines(keepends=True)})
    destination = ROOT / "notebooks/03_main_compression.ipynb"
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(notebook,indent=2)+"\n",encoding="utf-8")
    print(f"Fresh-kernel notebook: {destination}")


if __name__ == "__main__":
    build()
