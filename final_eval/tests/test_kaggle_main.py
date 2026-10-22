"""Fresh-kernel transfer and main orchestration tests; no GPU/model/API calls."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock
import zipfile

import pandas as pd

from final_eval import kaggle_main as runner
from final_eval import kaggle_stage_c_bootstrap as bootstrap
from final_eval.build_kaggle_main import cells


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / "dataset"
        self.source.mkdir()
        self.root = self.base / "working/symbiote"
        self.files = {"final_eval/example.py":b"# CRLF remains exact\r\nvalue=1\r\n",
                      "outputs/final_eval/main_gptoss_v1/cohort.csv":b"uid\nfixture\n",
                      "final_eval/kaggle_stage_c_bootstrap.py":Path(bootstrap.__file__).read_bytes()}
        self.manifest = {"bundle":bootstrap.BUNDLE,"files":{name:hashlib.sha256(data).hexdigest() for name,data in self.files.items()}}
        self.raw = (json.dumps(self.manifest,indent=2)+"\n").encode()
        self.digest = hashlib.sha256(self.raw).hexdigest()
        for name,data in self.files.items():
            path = self.source/name
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        (self.source/bootstrap.MANIFEST).write_bytes(self.raw)

    def zip_source(self):
        path = self.base/"symbiote_main_stage_c_v1.zip"
        with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
            for name,data in self.files.items(): archive.writestr(name,data)
            archive.writestr(bootstrap.MANIFEST,self.raw)
        return path

    def test_zip_and_expanded_install_idempotently_preserve_weights_and_source(self):
        model=self.root/"models/scorer.pt"; model.parent.mkdir(parents=True); model.write_bytes(b"retained-weights")
        old=self.root/"final_eval/example.py"; old.parent.mkdir(); old.write_bytes(b"earlier code")
        with contextlib.redirect_stdout(io.StringIO()):
            bootstrap.install(self.zip_source(),self.digest,self.root)
            bootstrap.install(self.source,self.digest,self.root)
        self.assertEqual(model.read_bytes(),b"retained-weights")
        self.assertEqual(old.read_bytes(),self.files["final_eval/example.py"])
        history=list((self.root/"outputs/final_eval/main_gptoss_v1/transfer_source_backups").rglob("example.py"))
        self.assertEqual(len(history),1)
        self.assertEqual(history[0].read_bytes(),b"earlier code")

    def test_bad_hash_or_changed_data_stops_before_overwriting_any_source(self):
        path=self.source/"final_eval/example.py"; path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"SHA256"):
            bootstrap.install(self.source,self.digest,self.root)
        self.assertFalse((self.root/"final_eval/example.py").exists())
        path.write_bytes(self.files["final_eval/example.py"])
        cohort=self.root/"outputs/final_eval/main_gptoss_v1/cohort.csv"
        cohort.parent.mkdir(parents=True); cohort.write_bytes(b"different data")
        with self.assertRaisesRegex(ValueError,"Existing artifact differs"):
            bootstrap.install(self.source,self.digest,self.root)
        self.assertEqual(cohort.read_bytes(),b"different data")
        self.assertFalse((self.root/"final_eval/example.py").exists())

    def test_actual_generated_first_three_cells_need_no_shared_variables(self):
        archive=self.zip_source()
        script_hash=hashlib.sha256(self.files["final_eval/kaggle_stage_c_bootstrap.py"]).hexdigest()
        blocks=cells(script_hash,self.digest)
        code=blocks[0].replace('Path("/kaggle/input")',f'Path({str(self.base)!r})')
        code=code.replace(f'namespace["install"](source, "{self.digest}")',
                          f'namespace["install"](source, "{self.digest}", root={str(self.root)!r})')
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code,"fresh_install_cell","exec"),{})
        self.assertEqual((self.root/"final_eval/example.py").read_bytes(),self.files["final_eval/example.py"])
        with patch("subprocess.run",return_value=Mock(returncode=0)) as command:
            for block in blocks[1:3]:
                fresh=block.replace('Path("/kaggle/working/symbiote")',f'Path({str(self.root)!r})')
                exec(compile(fresh,"independent_cell","exec"),{})
        self.assertEqual(command.call_count,2)
        self.assertEqual(command.call_args_list[0].args[0][-1],"setup")
        self.assertEqual(command.call_args_list[1].args[0][-1],"run")
        for block in blocks: compile(block,"notebook_cell","exec")

    def test_path_traversal_rejected(self):
        for name in ("../outside","/absolute","C:/outside","a\\b"):
            with self.assertRaises(ValueError): bootstrap.safe_relative(name)


class MainTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/"symbiote"; self.root.mkdir()
        previous=Path.cwd(); self.addCleanup(os.chdir,previous); os.chdir(self.root)
        patcher=patch.object(runner,"ROOT",self.root); patcher.start(); self.addCleanup(patcher.stop)
        runner.WORK.mkdir(parents=True)
        cohort=pd.DataFrame({"uid":[f"q{i}" for i in range(160)],"base_group_id":[f"g{i%125}" for i in range(160)]})
        cohort.to_csv(runner.WORK/"cohort.csv",index=False)
        runner.PILOT.parent.mkdir(parents=True)
        runner.PILOT.write_text(json.dumps({"passed":True,"pilot_base_groups":["pilot-family"]}))
        checkpoints={}
        for fold in range(5):
            path=runner.MODEL/f"fold{fold}/scorer.pt"; path.parent.mkdir(parents=True)
            path.write_bytes(f"fixture-weights-{fold}".encode())
            checkpoints[path.as_posix()]={"sha256":runner.sha(path),"bytes":path.stat().st_size}
        source=Path("final_eval/example.py"); source.parent.mkdir(); source.write_bytes(b"# source fixture\n")
        self.manifest={"files":{source.as_posix():runner.sha(source),
            (runner.WORK/"cohort.csv").as_posix():runner.sha(runner.WORK/"cohort.csv"),
            runner.PILOT.as_posix():runner.sha(runner.PILOT)},"checkpoints":checkpoints,"source_fingerprint":"fixture-source"}
        Path("MAIN_TRANSFER_MANIFEST.json").write_text(json.dumps(self.manifest))

    def rows(self):
        return pd.DataFrame([{"uid":f"q{i}","method":m,"target_ratio":ratio,
            "status":"ok","budget_compliant":True} for i in range(160) for m in runner.METHODS for ratio in (.5,.75)])

    def test_main_orchestration_exports_full_grid_without_model_or_api_work(self):
        version={"fixture":"version"}
        runner.save(runner.WORK/"setup_check.json",{"passed":True,"versions":version})
        def fake_execute(command,*args):
            self.assertIn("final_eval.batch",command)
            self.assertIn(str(runner.PILOT),[str(x) for x in command])
            self.rows().to_csv(runner.WORK/"compressed.csv",index=False)
        with patch.object(runner,"versions",return_value=version),patch.object(runner,"execute",side_effect=fake_execute), \
                patch("final_eval.utils.artifacts.source_fingerprint",return_value="fixture-source"),contextlib.redirect_stdout(io.StringIO()):
            report=runner.run_main()
            archive=runner.export_review()
        self.assertTrue(report["passed"]); self.assertEqual(report["conditions"],2240)
        with zipfile.ZipFile(archive) as saved:
            hashes=json.loads(saved.read("MAIN_REVIEW_MANIFEST.json"))
            self.assertIn((runner.WORK/"compressed.csv").as_posix(),hashes)
            self.assertFalse(any(n.endswith((".pt",".env")) for n in saved.namelist()))
            for name,expected in hashes.items(): self.assertEqual(hashlib.sha256(saved.read(name)).hexdigest(),expected)

    def test_missing_checkpoint_stops_and_failure_files_can_still_be_exported(self):
        (runner.MODEL/"fold2/scorer.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError,"do not retrain"):
            runner.verify_inputs()
        runner.save(runner.WORK/"failure_fixture.json",{"passed":False})
        with contextlib.redirect_stdout(io.StringIO()): archive=runner.export_review()
        with zipfile.ZipFile(archive) as saved:
            self.assertIn((runner.WORK/"failure_fixture.json").as_posix(),saved.namelist())

    def test_over_budget_and_incomplete_grids_rejected(self):
        frame=self.rows(); frame.loc[0,"budget_compliant"]=False
        frame.to_csv(runner.WORK/"compressed.csv",index=False)
        with self.assertRaisesRegex(ValueError,"grid failed"): runner.audit_outputs()
        frame.iloc[:-1].to_csv(runner.WORK/"compressed.csv",index=False)
        with self.assertRaisesRegex(ValueError,"grid failed"): runner.audit_outputs()

    def test_setup_restores_only_missing_packages_and_avoids_old_selective_context_pins(self):
        before={name:"installed" for name in runner.versions()}
        before.update(tiktoken=None,llmlingua=None,**{"selective-context":None})
        after={**before,"tiktoken":"0.14.0","llmlingua":"0.2.2","selective-context":"0.1.4"}
        with patch.object(runner,"versions",side_effect=[before,after]), \
                patch.object(runner,"execute") as calls,patch.object(runner,"verify_inputs"), \
                patch("importlib.util.find_spec",return_value=object()), contextlib.redirect_stdout(io.StringIO()):
            runner.setup()
        commands=[[str(x) for x in call.args[0]] for call in calls.call_args_list]
        installs=[c for c in commands if "pip" in c]
        self.assertEqual(len(installs),2)
        self.assertIn("tiktoken==0.14.0",installs[0])
        self.assertIn("--no-deps",installs[1])
        self.assertIn("selective-context==0.1.4",installs[1])
        self.assertFalse(any("train_scorer" in arg for c in commands for arg in c))
        self.assertTrue(json.loads((runner.WORK/"setup_check.json").read_text())["passed"])


if __name__ == "__main__": unittest.main()
