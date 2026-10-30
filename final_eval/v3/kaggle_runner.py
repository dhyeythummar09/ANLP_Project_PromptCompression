"""Kaggle setup/run/export in fresh child processes; notebook imports stay minimal."""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import venv
import zipfile

ROOT = Path(__file__).resolve().parents[2]
WORKING = ROOT.parent
LOGS = WORKING / "v3_logs"
RUN = ROOT / "outputs/v3/student_01"
CONFIG = ROOT / "final_eval/v3/configs/student.json"
STATUS = WORKING / "v3_status.json"
RESULT = WORKING / "symbiote_v3_results.zip"


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def redact(message):
    for name, value in os.environ.items():
        if value and len(value) >= 8 and (name.endswith("API_KEY") or name in {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"}):
            message = message.replace(value, "<redacted>")
    return message


def environment():
    env = os.environ.copy()
    cache = Path(tempfile.gettempdir()) / "symbiote_v3_cache"
    env.update(PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               USE_TF="0", USE_FLAX="0", HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0",
               HF_HOME=str(cache / "hf"), HF_HUB_CACHE=str(cache / "hf/hub"),
               HF_XET_CACHE=str(cache / "hf/xet"), TIKTOKEN_CACHE_DIR=str(cache / "tiktoken"),
               NLTK_DATA=str(cache / "nltk"), HF_HUB_DISABLE_XET="1")
    # Do not inherit a package overlay from another notebook experiment.
    env.pop("PYTHONPATH", None)
    return env


def execute(command, log_name, env):
    LOGS.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (LOGS / log_name).open("a", encoding="utf-8") as log:
        child = subprocess.Popen([str(x) for x in command], cwd=ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        stopped = threading.Event()

        def heartbeat():
            while not stopped.wait(30):
                print(f"{log_name}: running ({(time.monotonic() - started) / 60:.1f} min); saved work is retained.", flush=True)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for line in child.stdout:
                line = redact(line)
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = child.wait()
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
        finally:
            stopped.set()
            thread.join(timeout=1)
    if code:
        raise RuntimeError(f"{log_name} failed (exit {code}); inspect v3_logs/{log_name} and v3_status.json")


def setup(env):
    """Keep CUDA torch from Kaggle; all other installs live in an isolated overlay."""
    if sys.version_info < (3, 10):
        raise RuntimeError("Use a current Kaggle Python runtime (Python >=3.10)")
    execute([sys.executable, "-c", "import torch; assert torch.cuda.is_available(), 'Enable GPU T4 x2'; print('CUDA torch:',torch.__version__); print('GPU:',torch.cuda.get_device_name(0))"], "setup.log", env)
    location = Path(tempfile.gettempdir()) / "symbiote_v3_env"
    interpreter = location / "bin/python"
    if not interpreter.exists():
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(location)
    # The system pip is visible through system-site-packages, but targets this venv.
    torch_version = metadata.version("torch")
    constraints = LOGS / "torch_constraints.txt"
    constraints.write_text("torch==" + torch_version + "\n", encoding="utf-8")
    execute([interpreter, "-m", "pip", "install", "--disable-pip-version-check", "--only-binary=:all:",
             "-c", constraints, "-r", ROOT / "final_eval/v3/kaggle-requirements.txt"], "setup.log", env)
    check = "import torch; from transformers import AutoModel,AutoModelForTokenClassification,DistilBertModel; from llmlingua import PromptCompressor; import spacy,tiktoken,accelerate,nltk; assert torch.cuda.is_available(); assert torch.__version__ == " + repr(torch_version) + "; print('CUDA torch retained; real model/library imports passed')"
    execute([interpreter, "-c", check], "setup.log", env)
    language = "import importlib.util,subprocess,sys; available=importlib.util.find_spec('en_core_web_sm') is not None; subprocess.run([sys.executable,'-m','spacy','download','en_core_web_sm','--no-deps'],check=True) if not available else None"
    execute([interpreter, "-c", language], "setup.log", env)
    resources = "import os,nltk,spacy,tiktoken; spacy.load('en_core_web_sm'); tiktoken.get_encoding('cl100k_base'); [nltk.download(x,download_dir=os.environ['NLTK_DATA'],quiet=True,raise_on_error=True) for x in ('punkt','punkt_tab')]; from final_eval.v3.artifacts import versions; import json; print(json.dumps(versions()))"
    execute([interpreter, "-c", resources], "setup.log", env)
    snapshot = "import json,sys,platform,torch; from final_eval.v3.artifacts import versions,code_hash; from pathlib import Path; from final_eval.v3.kaggle_runner import save,LOGS; save(LOGS/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions(),cuda=torch.version.cuda,gpus=[dict(name=torch.cuda.get_device_name(i),bytes=torch.cuda.get_device_properties(i).total_memory) for i in range(torch.cuda.device_count())],source=code_hash()))"
    execute([interpreter, "-c", snapshot], "setup.log", env)
    execute([interpreter, "-m", "final_eval.v3.kaggle_runner", "weights"], "setup.log", env)
    execute([interpreter, "-B", "-m", "unittest", "final_eval.v3.tests.test_v3", "final_eval.v3.tests.test_kaggle", "-q"], "tests.log", env)
    execute([interpreter, "-m", "final_eval.v3", "preflight", "--config", CONFIG], "setup.log", env)
    return interpreter


def export(interpreter, env):
    """Export even after setup fails; do not depend on the notebook's final cell."""
    runtime = RUN / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    for source in LOGS.glob("*"):
        if source.is_file():
            shutil.copyfile(source, runtime / source.name)
    if STATUS.exists():
        shutil.copyfile(STATUS, runtime / STATUS.name)
    if interpreter is not None:
        execute([interpreter, "-m", "final_eval.v3", "bundle", "--config", CONFIG], "export.log", env)
        source = RUN.parent / (RUN.name + "_results.zip")
        shutil.copyfile(source, RESULT)
    else:
        # Dependency/setup failure: stdlib-only diagnostic archive, no API work.
        from final_eval.v3.kaggle_bootstrap import MANIFEST
        with zipfile.ZipFile(RESULT, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(LOGS.rglob("*")):
                if path.is_file():
                    archive.write(path, "runtime/" + path.relative_to(LOGS).as_posix())
            for path in [STATUS, CONFIG, ROOT / MANIFEST]:
                if path.is_file():
                    archive.write(path, path.name)
            for path in sorted(RUN.rglob("*")):
                if (path.is_file() and path.suffix not in {".lock", ".tmp", ".env"}
                        and path.name != "last.pt" and path != RUN / "initialization/common_initial.pt"):
                    archive.write(path, "run/" + path.relative_to(RUN).as_posix())
    print(f"RETURN THIS FILE: {RESULT}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["run", "weights"])
    parser.add_argument("--retry-failed", action="store_true", help="Explicitly retry reviewed failed requests; keeps the same cache/ledger")
    args = parser.parse_args(argv)
    if args.stage == "weights":
        from final_eval.v3.kaggle_bootstrap import fetch_weights
        fetch_weights(ROOT)
        return 0
    if os.name == "nt" or not Path("/kaggle/working").is_dir():
        raise RuntimeError("Run the generated notebook on Kaggle, not on the Windows laptop")
    os.chdir(ROOT)
    if not os.environ.get("JARVISLABS_API_KEY"):
        raise RuntimeError("Enable the JARVISLABS_API_KEY Kaggle secret before running")
    env, interpreter = environment(), None
    started = time.monotonic()
    state = {"success": False, "stage": "setup", "output_archive": str(RESULT),
             "note": "Success means execution completed, not an accuracy gain."}
    save(STATUS, state)
    try:
        interpreter = setup(env)
        state["stage"] = "pipeline"
        save(STATUS, state)
        command = [interpreter, "-u", "-m", "final_eval.v3", "run", "--config", CONFIG]
        if args.retry_failed:
            command.append("--retry-failed")
        execute(command, "pipeline.log", env)
        state.update(success=True, stage="completed")
    except BaseException as exc:
        state.update(error_type=type(exc).__name__, message=redact(str(exc)), traceback=redact(traceback.format_exc()))
        print(state["traceback"], flush=True)
    finally:
        state["wall_seconds"] = time.monotonic() - started
        save(STATUS, state)
        try:
            export(interpreter, env)
        except BaseException as exc:
            state.update(success=False, export_error=redact(str(exc)))
            save(STATUS, state)
            # The ordinary exporter can fail if installation was interrupted.
            export(None, env)
    return 0 if state["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
