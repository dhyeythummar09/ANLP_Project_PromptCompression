"""One Kaggle repair run: native whole-question LLMLingua, validate, merge, export.

Run on the retained Kaggle workspace, never the laptop. No target API requests,
training, new labels, weight deletion or regeneration of the six valid methods.
The original one-cell Markdown guide is preserved in the pre-cleanup archive.
See the root README.md for the completed repair's provenance and result paths.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

REPAIR_VERSION = 'llmlingua-whole-question-window-v2'
RUNNER_REVISION = 3
OLD_FILE_HASH = 'd9f811f73fba75907a90c0c668e269ec188db7ede632ec9798418ed61289aa2d'
NEW_FILE_HASH = 'a1e15840014a28d31412d4970e6276d45880718454d21f80b5ae09216f5b29aa'
OLD_SOURCE = '32b2877727148f214c86ab2802370f9ee6b0b611ddcd3561e9e31376b801d9dc'
OLD_RAW_FILE_HASH = 'eb4af11d9d1c4ea69097893b82322840bf2670e9c19164d5449e7eaa0cce70af'
COMPRESSOR_KEY = 'final_eval/compressor/compress.py'
OLD_SETTINGS = '            settings={"iterative_size":50} if method=="llmlingua" else {}'
NEW_SETTINGS = '''            settings = {}
            if method == "llmlingua":
                # A whole-question native window avoids both negative window
                # offsets on short inputs and legacy KV-cache reuse on TF5.
                # This policy is limited to questions fitting Phi-2's context.
                compressor = self.objects[method]
                length = compressor.get_token_length(text, add_special_tokens=False)
                if length > compressor.max_position_embeddings:
                    raise ValueError("LLMLingua whole-question window exceeds the model context; no silent truncation")
                settings = {"iterative_size": max(1, length)}'''
METHODS = ['ours', 'grouped_hybrid', 'grouped_signal', 'random', 'llmlingua_budgeted',
           'llmlingua2_budgeted', 'selective_context_budgeted']

# For the user's still-open notebook containing the previously supplied runner.
# This checks the entire original tree before updating its platform-specific
# expected hash. It is also tested end-to-end against that exact old runner.
CONTINUATION_CELL = r'''from pathlib import Path
import hashlib, os, shutil, tempfile
from final_eval.utils.artifacts import digest, file_hash, source_fingerprint

assert globals().get('REPAIR_VERSION') == 'llmlingua-whole-question-window-v2' and 'NEW_SOURCE' in globals(), 'Use this in the SAME notebook kernel as the failed repair cell.'
root = Path('/kaggle/working/symbiote')
os.chdir(root)
package = root / 'final_eval'
files = [package/'settings.py'] + sorted(package.glob('requirements*.txt'))
for name in ('compressor', 'inference', 'scorer', 'spans', 'utils', 'preparation'):
    files += sorted((package/name).rglob('*.py'))
components = {p.relative_to(root).as_posix(): file_hash(p) for p in files}
key = 'final_eval/compressor/compress.py'
normalized = (root/key).read_text(encoding='utf-8').encode()
assert hashlib.sha256(normalized).hexdigest() == 'a1e15840014a28d31412d4970e6276d45880718454d21f80b5ae09216f5b29aa', 'Compressor differs from the reviewed repair.'
original = {**components, key: 'eb4af11d9d1c4ea69097893b82322840bf2670e9c19164d5449e7eaa0cce70af'}
assert digest(original) == '32b2877727148f214c86ab2802370f9ee6b0b611ddcd3561e9e31376b801d9dc', 'Another core source file changed; stopped before GPU work.'

folder = Path('outputs/final_eval/pilot_llmlingua_window_v2')
history = folder/'failed_attempts'
history.mkdir(parents=True, exist_ok=True)
saved = Path(tempfile.mkdtemp(prefix='source_hash_failure_', dir=history))
for name in ('repair_summary.json', 'repair.json'):
    if (folder/name).is_file():
        shutil.copy2(folder/name, saved/name)
(saved/'cause.txt').write_text('Previous runner expected Windows CRLF raw bytes. Kaggle wrote equivalent LF Python code. Verified the exact repaired code and ALL other original source files before continuing.', encoding='utf-8')
NEW_SOURCE = source_fingerprint()
assert NEW_SOURCE == digest(components)
print('Verified repair and unchanged remaining source. Preserved previous failure:', saved)
main()
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def component_hashes(root):
    root = Path(root)
    package = root/'final_eval'
    files = [package/'settings.py'] + sorted(package.glob('requirements*.txt'))
    for directory in ('compressor', 'inference', 'scorer', 'spans', 'utils', 'preparation'):
        files += sorted((package/directory).rglob('*.py'))
    return {p.relative_to(root).as_posix(): sha(p) for p in files}


def tree_hash(components):
    data = json.dumps(components, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(data.encode()).hexdigest()


def reviewed_components(root):
    components = component_hashes(root)
    text = Path(root, COMPRESSOR_KEY).read_text(encoding='utf-8')
    if hashlib.sha256(text.encode()).hexdigest() not in {OLD_FILE_HASH, NEW_FILE_HASH}:
        raise ValueError('Compressor differs from both reviewed versions; no patch applied')
    original_tree = {**components, COMPRESSOR_KEY: OLD_RAW_FILE_HASH}
    if tree_hash(original_tree) != OLD_SOURCE:
        raise ValueError('A core file outside the reviewed comparator changed; no patch applied')
    return components


def validate_source_transition(before, after):
    without_compressor = lambda values: {k: v for k, v in values.items() if k != COMPRESSOR_KEY}
    if without_compressor(before) != without_compressor(after):
        raise ValueError('Repair unexpectedly changed another core source file')
    if after[COMPRESSOR_KEY] != NEW_FILE_HASH:
        raise ValueError('Repaired comparator bytes differ from the reviewed LF source')


def patch_source(path, audit):
    text = Path(path).read_text(encoding='utf-8')
    before = hashlib.sha256(text.encode()).hexdigest()
    if before == OLD_FILE_HASH:
        if text.count(OLD_SETTINGS) != 1:
            raise ValueError('Unexpected adapter source; refuse an ambiguous edit')
        write_text = text.replace(OLD_SETTINGS, NEW_SETTINGS)
        if hashlib.sha256(write_text.encode()).hexdigest() != NEW_FILE_HASH:
            raise ValueError('Repair source integrity check failed')
        backup = Path(audit, 'compress_before.py')
        if not backup.exists():
            backup.write_bytes(Path(path).read_bytes())
    elif before != NEW_FILE_HASH:
        raise ValueError('This is not the reviewed compressor source; original file left untouched')
    # Deterministic bytes on Windows and Linux, including a partially applied
    # repair from the previous failed runner. This does not alter Python logic.
    fixed = text.replace(OLD_SETTINGS, NEW_SETTINGS) if before == OLD_FILE_HASH else text
    Path(path).write_bytes(fixed.encode('utf-8'))
    result = {'policy': REPAIR_VERSION, 'observed_before_normalized_sha256': before,
              'after_normalized_sha256': NEW_FILE_HASH,
              'native_setting': 'iterative_size = max(1, number of Phi-2 input tokens)',
              'context_guard': 'raise before native compression if input exceeds model context',
              'budget_controller': 'official-controls-bracket-v1 unchanged',
              'cache_or_package_monkeypatch': False, 'output_truncation': False}
    write_json(Path(audit, 'repair.json'), result)
    return result


def run(command, log):
    print('Running:', ' '.join(map(str, command)), flush=True)
    with Path(log).open('a', encoding='utf-8') as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            stream.write(line)
            stream.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'Command exited {code}; full output saved in {log}')


def ensure_packages(folder):
    packages = ['torch', 'transformers', 'numpy', 'accelerate', 'tiktoken', 'nltk', 'pandas', 'scikit-learn']
    before = {}
    for package in packages:
        try:
            before[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    if not {'torch', 'transformers', 'pandas', 'scikit-learn'} <= before.keys():
        raise ValueError('The retained project environment is incomplete; restore its core dependencies first')
    try:
        version = metadata.version('llmlingua')
    except metadata.PackageNotFoundError:
        version = None
    if version != '0.2.2' or not set(packages) <= before.keys():
        constraints = folder / 'restore_constraints.txt'
        constraints.write_text(''.join(f'{k}=={v}\n' for k, v in before.items()), encoding='utf-8')
        run([sys.executable, '-m', 'pip', 'install', '-c', constraints, 'llmlingua==0.2.2'], folder/'setup.log')
    after = {p: metadata.version(p) for p in [*packages, 'llmlingua']}
    if any(after[p] != v for p, v in before.items()):
        raise ValueError('Dependency versions changed unexpectedly')
    write_json(folder/'repair_environment.json', {'python': sys.version, 'before': before, 'after': after})


def compression_check(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    keys = {(r['uid'], r['method'], float(r['target_ratio'])) for r in rows}
    questions = {r['uid'] for r in rows}
    expected = {(uid, method, ratio) for uid in questions for method in METHODS for ratio in (.75, .5)}
    bad = [r for r in rows if r['status'] != 'ok' or r['budget_compliant'].lower() not in ('true', '1', '1.0')
           or not math.isfinite(float(r['comp_tokens'] or 'inf'))
           or float(r['comp_tokens'] or 'inf') > float(r['budget_tokens'])]
    return {'passed': len(questions) == 16 and len(rows) == len(keys) == 224 and keys == expected and not bad,
            'conditions': len(rows), 'questions': len(questions), 'failed_or_over_budget': len(bad),
            'llmlingua_valid': sum(r['method'] == 'llmlingua_budgeted' and r not in bad for r in rows),
            'scope': 'Compression gate only. Target pilot/main inference not run.'}


def export_review(folder, original, model):
    destination = Path.cwd().parent
    number = 1
    while (destination/f'llmlingua_repair_review_{number:02d}.zip').exists():
        number += 1
    target = destination/f'llmlingua_repair_review_{number:02d}.zip'
    files = [p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.csv', '.json', '.log', '.py', '.txt'}]
    old_attempt = Path('outputs/final_eval/pilot_llmlingua_window_v2')
    if old_attempt.is_dir():
        files += [p for p in old_attempt.rglob('*') if p.is_file() and p.suffix in {'.json', '.log', '.py', '.txt'}]
    for name in ['cohort.csv', 'ours.csv', 'llmlingua2_budgeted.csv', 'selective_context_budgeted.csv']:
        path = original/name
        files.extend(p for p in (path, Path(str(path)+'.manifest.json')) if p.is_file())
    files += [p for p in (model/'training_metadata.json', model/'split.json') if p.is_file()]
    files.append(Path('final_eval/compressor/compress.py'))
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        manifest = {}
        for path in sorted(set(files)):
            archive.write(path, path.as_posix())
            manifest[path.as_posix()] = sha(path)
        archive.writestr('REPAIR_MANIFEST.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Review ZIP failed its CRC check')
    print('Download review ZIP:', target, flush=True)
    return str(target)


def main(root=None):
    supplied_root = root is not None  # Used by offline end-to-end fixture tests.
    root = Path(root) if supplied_root else Path('/kaggle/working/symbiote')
    if not root.is_dir() or (not supplied_root and os.name == 'nt'):
        raise RuntimeError('Run this repair on the retained Kaggle notebook, not the laptop')
    os.chdir(root)
    os.environ.update(HF_HUB_OFFLINE='0', TRANSFORMERS_OFFLINE='0',
                      HF_HOME='/kaggle/working/hf_cache', HF_HUB_CACHE='/kaggle/working/hf_cache/hub',
                      HF_XET_CACHE='/kaggle/working/hf_cache/xet')
    folder = Path('outputs/final_eval/pilot_llmlingua_repaired')
    folder.mkdir(parents=True, exist_ok=True)
    original = Path('outputs/final_eval/pilot')
    model = Path('models/span_scorer_v2_20261021')
    labels = Path('data/processed/team_v2_20261021/consensus_dataset.csv')
    summary = {'passed': False, 'repair_version': REPAIR_VERSION, 'runner_revision': RUNNER_REVISION,
               'target_api_calls': 0, 'training_jobs': 0}
    try:
        for path in [original/'cohort.csv', labels, model/'training_metadata.json', original/'ours.csv',
                     original/'llmlingua2_budgeted.csv', original/'selective_context_budgeted.csv']:
            if not path.is_file():
                raise FileNotFoundError(f'Required preserved artifact missing: {path}')
        if len(list(model.glob('fold*/scorer.pt'))) != 5:
            raise ValueError('Five scorer checkpoints must remain preserved; no retraining is performed')
        before_components = reviewed_components(root)
        before = tree_hash(before_components)
        ensure_packages(folder)
        run([sys.executable, '-c', 'import torch; from llmlingua import PromptCompressor; assert torch.cuda.is_available(), "Enable Kaggle GPU"; print("Imports/GPU ready:", torch.cuda.get_device_name(0))'], folder/'setup.log')
        patch_source(Path('final_eval/compressor/compress.py'), folder)
        after_components = reviewed_components(root)
        validate_source_transition(before_components, after_components)
        after = tree_hash(after_components)
        write_json(folder/'source_components.json', {'before': before_components, 'after': after_components})
        write_json(folder/'lineage.json', {'source_before': before, 'source_after': after,
                   'original_training_tree': OLD_SOURCE,
                   'source_validation': 'Exact reviewed comparator, LF bytes; all other core files unchanged',
                   'training_metadata_sha256': sha(model/'training_metadata.json'),
                   'training_source': json.loads((model/'training_metadata.json').read_text())['source'],
                   'scorer_retrained': False, 'change_scope': 'LLMLingua native window only',
                   'reused_stages': {str(original/name): sha(original/name) for name in
                                    ['ours.csv', 'llmlingua2_budgeted.csv', 'selective_context_budgeted.csv']}})
        run([sys.executable, '-u', '-m', 'final_eval.compressor.compress', '--input-csv', original/'cohort.csv',
             '--output-csv', folder/'llmlingua_budgeted.csv', '--model-dir', model, '--labels-file', labels,
             '--methods', 'llmlingua_budgeted', '--ratios', '.75', '.5'], folder/'compression.log')
        stages = [original/'ours.csv', folder/'llmlingua_budgeted.csv',
                  original/'llmlingua2_budgeted.csv', original/'selective_context_budgeted.csv']
        merged = folder/'compressed.csv'
        if merged.exists():
            recorded = json.loads(Path(str(merged)+'.merge.json').read_text())['inputs']
            if recorded != {str(p): sha(p) for p in stages}:
                raise ValueError('Existing repaired merge has different inputs; preserving it')
        else:
            run([sys.executable, '-m', 'final_eval.preparation.merge_compressions', '--inputs', *stages,
                 '--expected-methods', *METHODS, '--output', merged], folder/'merge.log')
        summary.update(compression_check(merged))
        write_json(folder/'compression_check.json', summary)
        if summary['passed']:
            if not (folder/'intrinsic').exists():
                run([sys.executable, '-m', 'final_eval.compressor.metrics', '--input', merged,
                     '--output-dir', folder/'intrinsic'], folder/'intrinsic.log')
            print('REPAIR PASS: LLMLingua 32/32; merged compression 224/224.', flush=True)
        else:
            print('REPAIR DID NOT PASS: failures preserved in review ZIP; no target calls.', flush=True)
    except Exception:
        summary.update(passed=False, setup_or_execution_traceback=traceback.format_exc())
        print(summary['setup_or_execution_traceback'], flush=True)
    finally:
        write_json(folder/'repair_summary.json', summary)
        export_review(folder, original, model)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    main()
