"""Offline orchestration checks; no model/provider execution."""
import csv
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from final_eval import repair_llmlingua as repair


class RepairRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reviewed_patch_is_exact_idempotent_and_preserves_old_source(self):
        current = Path('final_eval/compressor/compress.py').read_text(encoding='utf-8')
        original = current.replace(repair.NEW_SETTINGS, repair.OLD_SETTINGS)
        path = self.root/'compress.py'
        path.write_text(original, encoding='utf-8')
        repair.patch_source(path, self.root)
        self.assertEqual(path.read_text(), current)
        self.assertEqual((self.root/'compress_before.py').read_text(), original)
        repair.patch_source(path, self.root)
        self.assertEqual((self.root/'compress_before.py').read_text(), original)

    def test_unreviewed_source_is_not_edited(self):
        path = self.root/'compress.py'
        path.write_text('unexpected source')
        with self.assertRaisesRegex(ValueError, 'left untouched'):
            repair.patch_source(path, self.root)
        self.assertEqual(path.read_text(), 'unexpected source')

    def test_gate_requires_complete_successful_grid(self):
        path = self.root/'compressed.csv'
        rows = [{'uid': f'q{i}', 'method': method, 'target_ratio': ratio,
                 'status': 'ok', 'budget_compliant': 'True', 'comp_tokens': 8, 'budget_tokens': 10}
                for i in range(16) for method in repair.METHODS for ratio in (.75, .5)]
        def save():
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        save()
        self.assertTrue(repair.compression_check(path)['passed'])
        self.assertEqual(repair.compression_check(path)['llmlingua_valid'], 32)
        rows[0]['comp_tokens'] = 11
        save()
        self.assertFalse(repair.compression_check(path)['passed'])
        rows[0]['comp_tokens'] = 8
        rows.pop()
        save()
        self.assertFalse(repair.compression_check(path)['passed'])


class CrossPlatformRepairLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.repository = Path.cwd()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/'symbiote'
        self.root.mkdir()
        self.addCleanup(os.chdir, self.repository)
        for key in repair.component_hashes(self.repository):
            destination = self.root/key
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.repository/key, destination)
        # Replay the historical Kaggle tree, before later hosted-only controls.
        # Keep its strict source guard unchanged rather than accepting new code.
        client_fixture = self.repository/'final_eval/tests/fixtures/hosted_client_before_reasoning.py.txt'
        self.assertEqual(repair.sha(client_fixture), '22922fd857f8f5ec660e7182e0a443ee3fbbdc667affc74f79701eba69e3f929')
        (self.root/'final_eval/inference/client.py').write_bytes(client_fixture.read_bytes())
        current = (self.repository/repair.COMPRESSOR_KEY).read_text(encoding='utf-8')
        old = current.replace(repair.NEW_SETTINGS, repair.OLD_SETTINGS)
        self.compressor = self.root/repair.COMPRESSOR_KEY
        self.compressor.write_bytes(old.replace('\n', '\r\n').encode())
        self.original = self.root/'outputs/final_eval/pilot'
        self.original.mkdir(parents=True)
        (self.original/'cohort.csv').write_text('uid\n'+'\n'.join(f'q{i}' for i in range(16)))
        for name, methods in [('ours.csv', repair.METHODS[:4]),
                              ('llmlingua2_budgeted.csv', ['llmlingua2_budgeted']),
                              ('selective_context_budgeted.csv', ['selective_context_budgeted'])]:
            self.save_rows(self.original/name, self.rows(methods))
        labels = self.root/'data/processed/team_v2_20261021/consensus_dataset.csv'
        labels.parent.mkdir(parents=True); labels.write_text('fixture labels')
        model = self.root/'models/span_scorer_v2_20261021'
        for fold in range(5):
            path = model/f'fold{fold}/scorer.pt'
            path.parent.mkdir(parents=True); path.write_bytes(b'fixture, never loaded')
        (model/'training_metadata.json').write_text(json.dumps({'source': repair.OLD_SOURCE}))
        (model/'split.json').write_text('{}')
        failed = self.root/'outputs/final_eval/pilot_llmlingua_window_v2'
        failed.mkdir()
        (failed/'repair_summary.json').write_text('{"passed":false,"error":"Repaired source fingerprint mismatch"}')
        self.commands = []

    def rows(self, methods):
        return [dict(uid=f'q{i}', method=method, target_ratio=ratio, status='ok',
                     budget_compliant=True, comp_tokens=8, budget_tokens=10)
                for i in range(16) for method in methods for ratio in (.75, .5)]

    def save_rows(self, path, rows):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def fake_command(self, command, log):
        command = list(map(str, command)); self.commands.append(command)
        Path(log).write_text('offline simulated child, no models or network')
        if '-c' in command:
            return
        module = command[command.index('-m')+1]
        if module == 'final_eval.compressor.compress':
            self.save_rows(command[command.index('--output-csv')+1], self.rows(['llmlingua_budgeted']))
        elif module == 'final_eval.preparation.merge_compressions':
            paths = command[command.index('--inputs')+1:command.index('--expected-methods')]
            rows = []
            for path in paths:
                with open(path, newline='', encoding='utf-8') as stream:
                    rows.extend(csv.DictReader(stream))
            output = Path(command[command.index('--output')+1])
            self.save_rows(output, rows)
            Path(str(output)+'.merge.json').write_text(json.dumps({'inputs': {p: repair.sha(p) for p in paths}}))
        elif module == 'final_eval.compressor.metrics':
            Path(command[command.index('--output-dir')+1]).mkdir()
        else:
            self.fail(f'Unexpected command: {module}')

    def execute(self):
        with patch.dict(os.environ), patch.object(repair, 'ensure_packages'), \
                patch.object(repair, 'run', side_effect=self.fake_command), redirect_stdout(StringIO()):
            return repair.main(self.root)

    def test_full_runner_crlf_to_lf_and_resume_with_real_export(self):
        initial = repair.reviewed_components(self.root)
        self.assertEqual(repair.tree_hash(initial), repair.OLD_SOURCE)
        result = self.execute()
        self.assertTrue(result['passed']); self.assertEqual(result['llmlingua_valid'], 32)
        self.assertEqual(result['conditions'], 224)
        self.assertNotIn(b'\r\n', self.compressor.read_bytes())
        self.assertEqual(repair.sha(self.compressor), repair.NEW_FILE_HASH)
        self.assertTrue(self.execute()['passed'])
        self.assertEqual(len(list(self.root.parent.glob('llmlingua_repair_review_*.zip'))), 2)
        with zipfile.ZipFile(next(self.root.parent.glob('llmlingua_repair_review_*.zip'))) as archive:
            self.assertIsNone(archive.testzip())
            self.assertFalse(any(name.endswith('.pt') or name.endswith('.env') for name in archive.namelist()))
            self.assertIn('outputs/final_eval/pilot_llmlingua_window_v2/repair_summary.json', archive.namelist())
        (self.root/'outputs/final_eval/pilot_llmlingua_window_v2/repair_summary.json').read_text().index('fingerprint mismatch')
        self.assertFalse(any('train_scorer' in ' '.join(command) or 'compressor.evaluate' in command for command in self.commands))

    def test_resume_after_the_actual_reported_partial_patch(self):
        text = self.compressor.read_text(encoding='utf-8').replace(repair.OLD_SETTINGS, repair.NEW_SETTINGS)
        self.compressor.write_bytes(text.encode())  # Exactly Linux's first failed patch.
        self.assertEqual(repair.tree_hash(repair.reviewed_components(self.root)),
                         '244ec7e223d2c7e447f1cb9e551d47aeef99b5a6bdb791c57943ee03df9738c9')
        self.assertTrue(self.execute()['passed'])

    def test_exact_previous_cell_failure_then_short_notebook_continuation(self):
        old = (self.repository/'final_eval/tests/fixtures/failed_repair_runner_v2.py.txt').read_text(encoding='utf-8')
        old = old.replace("root = Path('/kaggle/working/symbiote')", f'root = Path({str(self.root)!r})')
        old = old.replace("if not root.is_dir() or os.name == 'nt':", 'if not root.is_dir():')
        namespace = {'__name__': 'legacy_notebook_fixture'}
        exec(compile(old, 'exact_previously_supplied_cell', 'exec'), namespace)
        namespace['run'] = self.fake_command
        namespace['ensure_packages'] = lambda folder: None
        namespace['export_review'] = repair.export_review
        write_text = Path.write_text
        def linux_write_text(path, text, *args, **kwargs):
            if path == Path('final_eval/compressor/compress.py'):
                return path.write_bytes(text.encode('utf-8'))
            return write_text(path, text, *args, **kwargs)
        actual_tree = lambda: repair.tree_hash(repair.component_hashes(self.root))
        with patch.dict(os.environ), patch.object(Path, 'write_text', linux_write_text), \
                patch('final_eval.utils.artifacts.source_fingerprint', side_effect=actual_tree), redirect_stdout(StringIO()):
            first = namespace['main']()
            self.assertFalse(first['passed'])
            self.assertIn('Repaired source fingerprint mismatch', first['setup_or_execution_traceback'])
            self.assertEqual(len(self.commands), 1)  # Setup only; no compression yet.
            continuation = repair.CONTINUATION_CELL.replace("root = Path('/kaggle/working/symbiote')", f'root = Path({str(self.root)!r})')
            exec(compile(continuation, 'short_notebook_continuation', 'exec'), namespace)
        folder = self.root/'outputs/final_eval/pilot_llmlingua_window_v2'
        result = json.loads((folder/'repair_summary.json').read_text())
        self.assertTrue(result['passed']); self.assertEqual(result['conditions'], 224)
        history = list((folder/'failed_attempts').glob('*/repair_summary.json'))
        self.assertEqual(len(history), 1)
        self.assertIn('Repaired source fingerprint mismatch', history[0].read_text())
        self.assertEqual(len(list(self.root.parent.glob('llmlingua_repair_review_*.zip'))), 2)

    def test_other_source_changes_are_still_rejected_before_any_process(self):
        path = self.root/'final_eval/settings.py'
        path.write_bytes(path.read_bytes()+b'\n# unexpected change\n')
        result = self.execute()
        self.assertFalse(result['passed'])
        self.assertIn('outside the reviewed comparator', result['setup_or_execution_traceback'])
        self.assertFalse(self.commands)


if __name__ == '__main__':
    unittest.main()
