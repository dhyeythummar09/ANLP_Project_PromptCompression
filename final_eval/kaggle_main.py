"""Fresh-process Kaggle main compression setup/run/export. No target API calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = Path("outputs/final_eval/main_gptoss_v1")
MODEL = Path("models/span_scorer_v2_20261021")
LABELS = Path("data/processed/team_v2_20261021/consensus_dataset.csv")
PILOT = Path("outputs/final_eval/target_pilots/gptoss/check.json")
METHODS = ["ours", "grouped_hybrid", "grouped_signal", "random", "llmlingua_budgeted",
           "llmlingua2_budgeted", "selective_context_budgeted"]
CORE = {"transformers":"5.16.1", "numpy":"2.1.3", "pandas":"2.3.3",
        "scikit-learn":"1.6.1", "spacy":"3.8.16", "accelerate":"1.14.0",
        "tiktoken":"0.14.0", "nltk":"3.9.1", "requests":None,
        "python-dotenv":None, "tqdm":None, "openai":None}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")


def versions():
    result = {}
    for name in ["torch", *CORE, "click", "llmlingua", "selective-context"]:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def configure_environment():
    cache = Path("/kaggle/working/hf_cache")
    os.environ.update(HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
        HF_HOME=str(cache), HF_HUB_CACHE=str(cache / "hub"), HF_XET_CACHE=str(cache / "xet"),
        TIKTOKEN_CACHE_DIR="/kaggle/working/tiktoken_cache", NLTK_DATA="/kaggle/working/nltk_data",
        TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")


def execute(arguments, log="run.log"):
    command = [str(x) for x in arguments]
    print("Running:", " ".join(command), flush=True)
    with (WORK / log).open("a", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", bufsize=1)
        stopped = threading.Event()
        def progress():
            while not stopped.wait(30):
                counts = []
                for name, expected in [("ours",1280), ("llmlingua_budgeted",320),
                                       ("llmlingua2_budgeted",320), ("selective_context_budgeted",320)]:
                    path = WORK / (name + ".csv")
                    if path.exists():
                        try:
                            with path.open(encoding="utf-8", newline="") as saved:
                                count = sum(1 for _ in csv.DictReader(saved))
                            counts.append(f"{name}:{count}/{expected}")
                        except (OSError, csv.Error):
                            pass
                print("Saved compression progress: " + (", ".join(counts) or "loading models / first output pending"), flush=True)
        if log == "run.log":
            threading.Thread(target=progress, daemon=True).start()
        try:
            for line in child.stdout:
                print(line, end="", flush=True)
                stream.write(line); stream.flush()
            code = child.wait()
        except BaseException:
            child.terminate(); child.wait()
            raise
        finally:
            stopped.set()
    if code:
        raise RuntimeError(f"Command failed ({code}); log preserved at {WORK / log}")


def verify_inputs():
    manifest = json.loads(Path("MAIN_TRANSFER_MANIFEST.json").read_text())
    for name, expected in manifest["files"].items():
        if not Path(name).is_file() or sha(name) != expected:
            raise ValueError(f"Installed input/code differs from transfer: {name}")
    for name, info in manifest["checkpoints"].items():
        path = Path(name)
        if not path.is_file():
            raise FileNotFoundError(f"Retained checkpoint missing: {path}. Restore the backed-up models directory; do not retrain.")
        if path.stat().st_size != info["bytes"] or sha(path) != info["sha256"]:
            raise ValueError(f"Retained checkpoint differs from verified training return: {path}")
    import pandas as pd
    cohort = pd.read_csv(WORK / "cohort.csv", keep_default_na=False)
    check = json.loads(PILOT.read_text())
    if len(cohort) != 160 or cohort.uid.nunique() != 160 or not check["passed"]:
        raise ValueError("Need the frozen160-question cohort and passed GPT-OSS target pilot")
    if set(cohort.base_group_id) & set(check["pilot_base_groups"]):
        raise ValueError("Main/pilot families overlap")
    from final_eval.utils.artifacts import source_fingerprint
    if source_fingerprint() != manifest["source_fingerprint"]:
        raise ValueError("Core source tree differs from the exact transferred bytes; preserve files and return review ZIP")
    print("Verified160 questions,125 families, five checkpoints, source and zero pilot overlap.", flush=True)


def setup():
    before = versions()
    save(WORK / "setup_versions_before.json", before)
    if before["torch"] is None:
        raise RuntimeError("Kaggle PyTorch runtime missing; select a GPU runtime before setup")
    # Preserve installed core packages. Restore missing wheels, never the legacy full requirements stack.
    constraints = WORK / "restore_constraints.txt"
    constraints.write_text("".join(f"{name}=={version}\n" for name, version in before.items()
                                   if version and name not in {"llmlingua", "selective-context"}), encoding="utf-8")
    missing = [name + ("==" + version if version else "") for name, version in CORE.items() if before[name] is None]
    if before["llmlingua"] != "0.2.2":
        missing.append("llmlingua==0.2.2")
    if missing:
        execute([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "-c", constraints, *missing], "setup.log")
    if before["selective-context"] != "0.1.4":
        execute([sys.executable, "-m", "pip", "install", "--only-binary=:all:", "--no-deps", "selective-context==0.1.4"], "setup.log")
    after = versions()
    if any(after[name] != value for name, value in before.items()
           if value is not None and name not in {"llmlingua", "selective-context"}):
        raise ValueError("An installed core package changed unexpectedly; stop before main compression")
    save(WORK / "setup_versions_after.json", after)
    execute([sys.executable, "-c", "import torch; from llmlingua import PromptCompressor; from selective_context import SelectiveContext; import spacy,tiktoken; assert torch.cuda.is_available(), 'Enable Kaggle GPU'; print('GPU:', torch.cuda.get_device_name(0)); print('Imports passed')"], "setup.log")
    # The language package is installed in a child, so no notebook restart/import-state assumption.
    import importlib.util
    if importlib.util.find_spec("en_core_web_sm") is None:
        execute([sys.executable, "-m", "spacy", "download", "en_core_web_sm", "--no-deps"], "setup.log")
    execute([sys.executable, "-c", "import os,nltk,spacy,tiktoken; spacy.load('en_core_web_sm'); print('Tokenizer:',len(tiktoken.get_encoding('cl100k_base').encode('Readiness check'))); [nltk.download(name,download_dir=os.environ['NLTK_DATA'],quiet=True,raise_on_error=True) for name in ('punkt','punkt_tab')]; print('Language resources ready')"], "setup.log")
    verify_inputs()
    execute([sys.executable, "-m", "final_eval.handoff", "environment", "--output", WORK / "environment.json"], "setup.log")
    execute([sys.executable, "-m", "final_eval.compressor.compress", "--input-csv", WORK / "cohort.csv",
             "--output-csv", WORK / "ours.csv", "--model-dir", MODEL, "--labels-file", LABELS,
             "--methods", *METHODS, "--ratios", ".75", ".5", "--dry-run"], "setup.log")
    save(WORK / "setup_check.json", {"passed":True, "versions":after,
        "source_fingerprint":json.loads(Path("MAIN_TRANSFER_MANIFEST.json").read_text())["source_fingerprint"],
        "target_api_calls":0, "training_jobs":0})
    print("SETUP PASS. Run the MAIN COMPRESSION cell next.", flush=True)


def audit_outputs():
    import pandas as pd
    from final_eval.utils.artifacts import parse_bool, require_unique
    frame = pd.read_csv(WORK / "compressed.csv", keep_default_na=False)
    cohort = pd.read_csv(WORK / "cohort.csv", keep_default_na=False)
    require_unique(frame, ["uid", "method", "target_ratio"])
    expected = {(uid, method, ratio) for uid in cohort.uid for method in METHODS for ratio in (.5, .75)}
    actual = set(frame[["uid", "method", "target_ratio"]].itertuples(index=False, name=None))
    failed = int(frame.status.ne("ok").sum())
    over = int((~frame.budget_compliant.map(parse_bool)).sum())
    report = {"passed":actual == expected and failed == 0 and over == 0,
        "questions":int(frame.uid.nunique()), "conditions":len(frame), "expected_conditions":2240,
        "failed":failed, "over_budget":over, "target_api_calls":0, "training_jobs":0,
        "compressed_sha256":sha(WORK / "compressed.csv"), "scope":"Main compression only; downstream main inference not run"}
    save(WORK / "compression_check.json", report)
    if not report["passed"]:
        raise ValueError("Main compression grid failed; return review ZIP before API calls. No automatic trimming/relabeling.")
    print("MAIN COMPRESSION PASS:2240/2240. Download the review ZIP; next stage is laptop API evaluation.", flush=True)
    return report


def run_main():
    ready = WORK / "setup_check.json"
    if not ready.exists() or not json.loads(ready.read_text())["passed"]:
        raise ValueError("Run SETUP cell successfully first; variables from old kernels are not needed")
    if json.loads(ready.read_text())["versions"] != versions():
        raise ValueError("Packages changed since setup; rerun SETUP before main")
    verify_inputs()
    execute([sys.executable, "-u", "-m", "final_eval.batch", "--cohort", WORK / "cohort.csv",
             "--work-dir", WORK, "--model-dir", MODEL, "--labels-file", LABELS, "--pilot-check", PILOT])
    return audit_outputs()


def export_review():
    number = 1
    while (ROOT.parent / f"symbiote_main_review_{number:02d}.zip").exists():
        number += 1
    target = ROOT.parent / f"symbiote_main_review_{number:02d}.zip"
    files = {p.relative_to(ROOT).as_posix():p for p in (ROOT / WORK).rglob("*")
             if p.is_file() and p.suffix in {".csv", ".json", ".log", ".txt", ".py"}}
    files[PILOT.as_posix()] = ROOT / PILOT
    files["MAIN_TRANSFER_MANIFEST.json"] = ROOT / "MAIN_TRANSFER_MANIFEST.json"
    manifest = json.loads((ROOT / "MAIN_TRANSFER_MANIFEST.json").read_text())
    for name in manifest["files"]:
        if name.startswith("final_eval/") and name.endswith(".py"):
            files["source_snapshot/" + name] = ROOT / name
    hashes = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            if path.is_file():
                archive.write(path, name); hashes[name] = sha(path)
        archive.writestr("MAIN_REVIEW_MANIFEST.json", json.dumps(hashes, indent=2))
    print(f"Download review ZIP: {target} ({target.stat().st_size / 1024**2:.2f} MiB)", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["setup", "run", "export"])
    args = parser.parse_args()
    if os.name == "nt" or not Path("/kaggle/working").exists():
        raise SystemExit("Run this module on Kaggle, not on the Windows laptop")
    os.chdir(ROOT); WORK.mkdir(parents=True, exist_ok=True); configure_environment()
    started = time.monotonic()
    passed = False
    try:
        {"setup":setup, "run":run_main, "export":lambda:None}[args.stage]()
        passed = True
    except BaseException:
        save(WORK / f"failure_{args.stage}_{time.time_ns()}.json", {"stage":args.stage,
            "traceback":traceback.format_exc(), "target_api_calls":0, "training_jobs":0})
        traceback.print_exc()
    finally:
        save(WORK / f"last_{args.stage}_attempt.json", {"passed":passed,"wall_seconds":time.monotonic()-started,
            "target_api_calls":0,"training_jobs":0,"stage":args.stage})
        export_review()
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
