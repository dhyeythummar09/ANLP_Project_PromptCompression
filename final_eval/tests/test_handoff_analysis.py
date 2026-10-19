"""Offline analysis and execution gates; all inputs and generations are fabricated."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import pandas as pd
from unittest.mock import Mock
from final_eval.inference.client import HostedClient
from final_eval.analysis import agreement, feature_baseline, results
from final_eval.compressor import evaluate, compress
from final_eval.compressor.budget import TokenCounter
from final_eval.preparation import cohort
from final_eval.preparation.schema import prepare_problems, normalize_votes, build_consensus, grouped_folds
from final_eval.scorer.data import build_span_table
from final_eval.utils.artifacts import file_hash
from final_eval.utils.profiling import profile_cli
from final_eval import handoff
from test_cli_pipeline import FakeTarget


class AnalysisHandoffTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup); self.root=Path(temp.name)
        problems=[]; votes=[]
        for i in range(12):
            prompt=f'Person {i} has 5 apples. How many apples?'
            base=dict(example_id=str(i),source_dataset='gsm8k',original_prompt=prompt,ground_truth_answer='5')
            problems.append(base); start=prompt.index('5')
            for model in range(3):
                critical=model<i%4
                votes.append({**base,'span_text':'5','span_start':start,'span_end':start+1,'span_category':'number',
                    'labeling_model':f'm{model}','model_answer_original':'Final Answer: 5',
                    'model_answer_perturbed':'Final Answer: 0' if critical else 'Final Answer: 5'})
        self.problems=prepare_problems(pd.DataFrame(problems))
        self.votes=normalize_votes(pd.DataFrame(votes),self.problems,regrade=True)
        self.consensus=build_consensus(self.votes)
        self.prepared=self.root/'prepared'; self.prepared.mkdir()
        self.consensus_file=self.prepared/'consensus_dataset.csv'; self.consensus.to_csv(self.consensus_file,index=False)
        self.votes_file=self.prepared/'per_model_labels.csv'; self.votes.to_csv(self.votes_file,index=False)
        self.problem_file=self.prepared/'problems.csv'; self.problems.to_csv(self.problem_file,index=False)
        self.model=self.root/'model'; self.model.mkdir()
        table=build_span_table(self.consensus)
        partitions={str(f):{role:sorted(uids) for role,uids in roles.items()} for f,roles in grouped_folds(table,3,.2)}
        folds={u:int(f) for f,roles in partitions.items() for u in roles['test']}
        (self.model/'split.json').write_text(json.dumps({'protocol':'symbiote-v2','partitions':partitions,'folds':folds,
            'base_groups':dict(zip(table.uid,table.base_group_id))}))
        (self.model/'training_metadata.json').write_text(json.dumps({'consensus_sha256':file_hash(self.consensus_file)}))
        neural=table.drop(columns='original_prompt').copy(); neural['prob']=.6; neural['prior_prob']=.4
        neural.iloc[:-1].to_csv(self.model/'scorer_oof_predictions.csv',index=False)

    def test_agreement_handles_prevalence_and_shared_occurrences(self):
        target=self.root/'agreement'; agreement.run(self.votes_file,target)
        report=pd.read_csv(target/'pairwise.csv')
        row=report[(report.source_dataset=='ALL')&(report.model_a=='m0')&(report.model_b=='m1')].iloc[0]
        self.assertEqual(row.n_common,12)
        self.assertAlmostEqual(row.agreement,.75)
        self.assertAlmostEqual(row.positive_agreement,.8)
        self.assertTrue((target/'agreement.png').exists())
        stats=agreement.pair_statistics(pd.Series([0,0]),pd.Series([0,0]))
        self.assertIsNone(stats['cohen_kappa']); self.assertIsNone(stats['positive_agreement'])

    def test_feature_baseline_shared_support_and_no_test_target_fit(self):
        out=self.root/'feature'; feature_baseline.run(self.consensus_file,self.model,out)
        report=json.loads((out/'feature_report.json').read_text())
        self.assertEqual(report['n_feature_oof'],12); self.assertEqual(report['n_shared_with_neural'],11)
        before=pd.read_csv(out/'feature_oof.csv')
        split=json.loads((self.model/'split.json').read_text()); held=set(split['partitions']['0']['test'])
        changed=self.consensus.copy(); mask=changed.uid.isin(held)
        changed.loc[mask,'model_specific_label']='non_critical'; changed.loc[mask,'agreement_score']=0.
        changed.to_csv(self.consensus_file,index=False)
        (self.model/'training_metadata.json').write_text(json.dumps({'consensus_sha256':file_hash(self.consensus_file)}))
        other=self.root/'other_feature'; feature_baseline.run(self.consensus_file,self.model,other)
        after=pd.read_csv(other/'feature_oof.csv')
        self.assertEqual(before[before.fold==0].feature_prob.tolist(),after[after.fold==0].feature_prob.tolist())
        split['partitions']['0']['train']+=split['partitions']['0']['test']
        (self.model/'split.json').write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError,'leakage'):
            feature_baseline.run(self.consensus_file,self.model,self.root/'bad_feature')

    def _evaluated_fixture(self):
        rows=[]
        for row in self.problems.to_dict('records'):
            for method,prompt in [('grouped_hybrid','5 apples'),('random','missing facts')]:
                rows.append({**row,'method':method,'target_ratio':.5,'actual_ratio':.5,'compressed_prompt':prompt,
                    'status':'ok','budget_compliant':True,'orig_tokens':20,'crit_total':1,
                    'crit_kept_lower':int(method=='grouped_hybrid'),'crit_kept_upper':int(method=='grouped_hybrid'),
                    'latency_ms':1.})
        path=self.root/'compressed.csv'; pd.DataFrame(rows).to_csv(path,index=False)
        cfg=self.root/'target.json'; cfg.write_text(json.dumps({'name':'fixture','backend':'local'}))
        args=SimpleNamespace(model_config=str(cfg),input_csv=str(path),output_csv=str(self.root/'responses.csv'),
            methods=None,ratios=None,limit=None,dry_run=False,retry_failed=False,report_dir=str(self.root/'reports'),
            focus='grouped_hybrid',bootstrap=100)
        with patch.object(evaluate,'make_client',return_value=FakeTarget()),redirect_stdout(StringIO()):
            evaluate.run_eval(args); evaluate.summarize(args)
        return args

    def test_results_reports_and_family_disjoint_prediction(self):
        args=self._evaluated_fixture(); target=self.root/'analysis'
        results.run(args.input_csv,args.output_csv,args.report_dir,target)
        prediction=pd.read_csv(target/'retention_prediction.csv')
        self.assertEqual(prediction.status.iloc[0],'ok')
        oof=pd.read_csv(target/'retention_prediction_oof.csv')
        self.assertTrue(oof.groupby('base_group_id').analysis_fold.nunique().eq(1).all())
        ledger=pd.read_csv(target/'claims_ledger.csv')
        self.assertEqual(ledger.evidence_status.iloc[0],'positive_paired_interval')
        self.assertTrue((target/'accuracy_tokens_0.png').exists())

    def test_pilot_gate_blocks_missing_conditions_and_budget_overflow(self):
        args=self._evaluated_fixture()
        with redirect_stdout(StringIO()):
            good=handoff.check(args.input_csv,args.output_csv,self.root/'gate.json')
        self.assertTrue(good['passed'])
        comp=pd.read_csv(args.input_csv); comp.loc[0,'budget_compliant']=False; comp.to_csv(args.input_csv,index=False)
        res=pd.read_csv(args.output_csv).iloc[1:]; res.to_csv(args.output_csv,index=False)
        with redirect_stdout(StringIO()):
            bad=handoff.check(args.input_csv,args.output_csv,self.root/'bad_gate.json')
        self.assertFalse(bad['passed']); self.assertEqual(bad['budget_violations'],1)

    def test_pilot_families_excluded_from_main_cohort(self):
        pilot=self.root/'pilot.csv'; main=self.root/'main.csv'
        common=['cohort','--problems',str(self.problem_file),'--split',str(self.model/'split.json')]
        with patch('sys.argv',common+['--output',str(pilot),'--max-questions','2']),redirect_stdout(StringIO()):
            cohort.main()
        with patch('sys.argv',common+['--output',str(main),'--exclude-cohort',str(pilot),'--max-questions','10']),redirect_stdout(StringIO()):
            cohort.main()
        a=pd.read_csv(pilot); b=pd.read_csv(main)
        self.assertFalse(set(a.base_group_id)&set(b.base_group_id)); self.assertEqual(len(b),10)

    def test_archive_excludes_env_weights_and_rejects_inline_key(self):
        run=self.root/'run'; run.mkdir(); (run/'results.csv').write_text('n\n1\n')
        (run/'.env').write_text('secret'); (self.model/'scorer.pt').write_bytes(b'fake weights')
        archive=self.root/'handoff.zip'
        with patch('final_eval.handoff.Path.cwd',return_value=self.root),redirect_stdout(StringIO()):
            handoff.bundle(run,self.prepared,self.model,archive)
        with zipfile.ZipFile(archive) as saved:
            self.assertFalse(any(name.endswith('.env') or name.endswith('.pt') for name in saved.namelist()))
            self.assertIn('HANDOFF_MANIFEST.json',saved.namelist())
        (run/'target.json').write_text(json.dumps({'backend':'hosted','api_key':'fixture-secret'}))
        with patch('final_eval.handoff.Path.cwd',return_value=self.root),self.assertRaisesRegex(ValueError,'credentials'):
            handoff.bundle(run,self.prepared,self.model,self.root/'bad.zip')
        self.assertFalse((self.root/'bad.zip').exists())

    def test_process_profile_does_not_write_during_dry_run(self):
        output=self.root/'out.csv'; output.write_text('fixture')
        with patch('sys.argv',['fixture','--output-csv',str(output),'--dry-run']):
            profile_cli(lambda:None,'--output-csv')
        self.assertFalse(Path(str(output)+'.profiles').exists())
        with patch('sys.argv',['fixture','--output-csv',str(output)]):
            profile_cli(lambda:None,'--output-csv')
        self.assertEqual(len(list(Path(str(output)+'.profiles').glob('*.json'))),1)

    def test_deletion_check_completed_resume_makes_no_calls(self):
        cfg=self.root/'deletion_target.json'; cfg.write_text(json.dumps({'name':'fixture','backend':'local'}))
        args=SimpleNamespace(model_config=str(cfg),input_csv=str(self.votes_file),n=2,seed=42,dry_run=False,
            output_csv=str(self.root/'deletion.csv'))
        fake=FakeTarget()
        with patch.object(evaluate,'make_client',return_value=fake),redirect_stdout(StringIO()):
            evaluate.deletion_check(args); calls=fake.calls; evaluate.deletion_check(args)
        self.assertEqual(fake.calls,calls)
        self.assertEqual(json.loads(Path(args.output_csv+'.summary.json').read_text())['n_attempted'],2)

    @patch('final_eval.inference.client.requests.post')
    def test_raising_request_ceiling_keeps_existing_attempt_ledger(self,post):
        response=Mock(ok=True,status_code=200)
        response.json.return_value={'model':'fixture','choices':[{'message':{'content':'Final Answer: 5'},'finish_reason':'stop'}]}
        post.return_value=response
        cfg={'name':'fixture','base_url':'https://example.invalid/v1','request_budget':1,'sleep_seconds':0}
        ledger=self.root/'usage.json'
        client=HostedClient(cfg,usage_path=ledger)
        self.assertEqual(client.generate('one').status,'ok')
        self.assertEqual(client.generate('two').status,'request_budget_exhausted')
        resumed=HostedClient({**cfg,'request_budget':2},usage_path=ledger)
        self.assertEqual(resumed.generate('two').status,'ok')
        self.assertEqual(post.call_count,2)
        saved=json.loads(ledger.read_text()); self.assertEqual(saved['attempts'],2)
        self.assertEqual(len(saved['execution_history']),2)

    def test_main_guard_rejects_pilot_overlap_before_model_loading(self):
        args=self._evaluated_fixture(); gate=self.root/'pilot_gate.json'
        with redirect_stdout(StringIO()):
            handoff.check(args.input_csv,args.output_csv,gate)
        args.pilot_check=str(gate)
        with patch.object(evaluate,'make_client',side_effect=AssertionError('must not initialize')):
            with self.assertRaisesRegex(ValueError,'overlap'):
                evaluate.run_eval(args)

    def test_new_request_ceiling_does_not_regenerate_completed_conditions(self):
        args=self._evaluated_fixture()
        cfg=json.loads(Path(args.model_config).read_text()); cfg['request_budget']=3000
        Path(args.model_config).write_text(json.dumps(cfg))
        with patch.object(evaluate,'make_client',side_effect=AssertionError('completed responses must be reused')),redirect_stdout(StringIO()):
            evaluate.run_eval(args)

    def test_offline_reporting_rejects_stale_summaries_without_inference(self):
        args=self._evaluated_fixture()
        frame=pd.read_csv(args.output_csv); frame.loc[0,'response']='edited fixture response'; frame.to_csv(args.output_csv,index=False)
        with self.assertRaisesRegex(ValueError,'stale inputs'):
            results.run(args.input_csv,args.output_csv,args.report_dir,self.root/'stale_analysis')

    def test_external_budget_controller_keeps_native_and_only_selects_official_outputs(self):
        runner=compress.BaselineRunner('cpu','fixture1','fixture2'); counter=TokenCounter('regex-test-only')
        def native(method,text,ratio,budget):
            return ' '.join(['word']*min(20,budget+2))
        with patch.object(runner,'run',side_effect=native):
            text,status,details=runner.run_budgeted('llmlingua2_budgeted','fixture',.5,10,counter)
        attempts=json.loads(details['budget_search_json'])
        self.assertEqual(status,'ok'); self.assertLessEqual(counter(text),10)
        self.assertEqual(details['native_comp_tokens'],12)
        self.assertIn(text,[a['text'] for a in attempts])
        self.assertEqual(counter(text),max(a['tokens'] for a in attempts if a['tokens']<=10 and a['text']))
        with patch.object(runner,'run',return_value=' '.join(['word']*20)):
            text,status,_=runner.run_budgeted('llmlingua_budgeted','fixture',.5,10,counter)
        self.assertEqual(status,'baseline_budget_infeasible'); self.assertEqual(counter(text),20)


if __name__=='__main__':
    unittest.main()
