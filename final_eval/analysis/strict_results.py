"""Full-cohort end-to-end accuracy: retain provider failures as incorrect.

Raw response_status is never changed or labeled successful. Existing complete-
response summaries can be used separately as a sensitivity analysis.
"""
import argparse
from pathlib import Path

import pandas as pd

from final_eval.utils.artifacts import atomic_csv, atomic_json, claim_output, file_hash, parse_bool, require_unique
from final_eval.utils.statistics import cluster_ci

KEY = ['uid','method','target_ratio']


def strict_frame(comp, res):
    require_unique(comp,KEY,'compression')
    require_unique(res,KEY+['eval_model'],'responses')
    if not comp.status.eq('ok').all() or not comp.budget_compliant.map(parse_bool).all():
        raise ValueError('Expected passing budget-compliant compression')
    expected = set(comp[KEY].itertuples(index=False,name=None)) | {(u,'none',1.0) for u in comp.uid.unique()}
    for model, group in res.groupby('eval_model'):
        if set(group[KEY].itertuples(index=False,name=None)) != expected:
            raise ValueError(f'Incomplete recorded condition grid for {model}')
    frame = res.copy()
    frame['strict_correct'] = [int(parse_bool(r.correct)) if r.response_status == 'ok' else 0
                               for r in frame.itertuples()]
    return frame


def run(compressed, responses, output_dir, bootstrap=1000):
    comp = pd.read_csv(compressed,keep_default_na=False)
    res = pd.concat([pd.read_csv(p,keep_default_na=False) for p in responses],ignore_index=True)
    frame = strict_frame(comp,res)
    target = Path(output_dir)
    claim_output(target)
    target.mkdir(parents=True)
    summary=[]; coverage=[]; paired=[]
    for model, group in frame.groupby('eval_model'):
        originals = group[group.method.eq('none')].set_index('uid')
        for (ratio,method), values in group[~group.method.eq('none')].groupby(['target_ratio','method']):
            values=values.sort_values('uid')
            actual=comp[comp.method.eq(method)&comp.target_ratio.eq(ratio)].set_index('uid')
            for source,subset in [('ALL',values)]+list(values.groupby('source_dataset')):
                ci=cluster_ci(subset.strict_correct,subset.base_group_id,bootstrap)
                full=originals.loc[subset.uid,'strict_correct'].to_numpy()
                outcomes=subset.strict_correct.to_numpy()
                summary.append(dict(eval_model=model,target_ratio=ratio,method=method,source_dataset=source,
                    n=len(subset),n_groups=subset.base_group_id.nunique(),accuracy=float(outcomes.mean()),
                    ci_lo=ci[0],ci_hi=ci[1],full_input_accuracy=float(full.mean()),
                    conditional_preservation=float(outcomes[full==1].mean()) if (full==1).any() else None,
                    actual_ratio=float(actual.loc[subset.uid,'actual_ratio'].mean())))
            coverage.append(dict(eval_model=model,target_ratio=ratio,method=method,n_planned=len(values),
                n_provider_success=int(values.response_status.eq('ok').sum()),
                n_provider_failure=int(values.response_status.ne('ok').sum()),
                n_parse_ok=int(values.parse_status.eq('ok').sum()),n_primary=len(values)))
        for ratio, values in group[~group.method.eq('none')].groupby('target_ratio'):
            pivot=values.pivot(index='uid',columns='method',values='strict_correct').sort_index()
            meta=values.drop_duplicates('uid').set_index('uid').loc[pivot.index]
            for other in sorted(set(pivot)-{'grouped_hybrid'}):
                difference=pivot.grouped_hybrid-pivot[other]
                ci=cluster_ci(difference,meta.base_group_id,bootstrap)
                paired.append(dict(eval_model=model,target_ratio=ratio,focus='grouped_hybrid',other=other,
                    n_paired=len(pivot),n_groups=meta.base_group_id.nunique(),difference=float(difference.mean()),
                    ci_lo=ci[0],ci_hi=ci[1],policy='All planned questions; provider/parse failures count incorrect; unadjusted family-bootstrap interval'))
    accuracy=pd.DataFrame(summary)
    atomic_csv(accuracy,target/'accuracy.csv')
    atomic_csv(pd.DataFrame(coverage),target/'coverage.csv')
    atomic_csv(pd.DataFrame(paired),target/'paired.csv')
    atomic_csv(accuracy[accuracy.source_dataset.ne('ALL')].groupby(['eval_model','method','target_ratio'],as_index=False)
        .agg(dataset_macro_accuracy=('accuracy','mean'),n_datasets=('source_dataset','nunique')),target/'macro_accuracy.csv')
    atomic_csv(frame.groupby(['eval_model','method','target_ratio','response_status','parse_status'],as_index=False)
        .agg(n=('uid','size')),target/'grading_coverage.csv')
    atomic_json(target/'manifest.json',{'compression_sha256':file_hash(compressed),
        'response_files':{str(p):file_hash(p) for p in responses},'bootstrap':bootstrap,'seed':42,
        'policy':'Strict full planned cohort: provider failures and invalid parses count incorrect. Raw provider statuses unchanged. Technical completion gate may remain failed.',
        'clustering':'base_group_id','multiple_testing':'Unadjusted95% intervals; no family-wise correction',
        'new_api_calls':0})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    overall=accuracy[accuracy.source_dataset.eq('ALL')]
    models=sorted(overall.eval_model.unique())
    fig,axes=plt.subplots(1,len(models),figsize=(6*len(models),4),squeeze=False,sharey=True)
    for axis,model in zip(axes[0],models):
        for method,points in overall[overall.eval_model.eq(model)].groupby('method'):
            points=points.sort_values('actual_ratio')
            axis.plot(points.actual_ratio,points.accuracy,marker='o',label=method)
        axis.set(xlabel='Mean actual retained-token fraction',ylabel='Strict full-cohort accuracy',ylim=(0,1),title=model)
        axis.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(target/'two_reader_accuracy.png',dpi=180)
    plt.close(fig)
    print(overall.pivot(index='method',columns=['eval_model','target_ratio'],values='accuracy').to_string())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compressed',required=True)
    parser.add_argument('--responses',nargs='+',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--bootstrap',type=int,default=1000)
    args=parser.parse_args()
    run(args.compressed,args.responses,args.output_dir,args.bootstrap)


if __name__=='__main__':
    main()
