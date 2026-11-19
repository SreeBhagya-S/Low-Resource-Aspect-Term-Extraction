#!/usr/bin/env python3
"""Summarize ALL pre-registered repeat metrics and verify exact test predictions."""
import argparse,ast,json,hashlib
from pathlib import Path
import pandas as pd

def main():
  ap=argparse.ArgumentParser(description=__doc__)
  ap.add_argument('--plan',required=True);ap.add_argument('--test-manifest',required=True)
  ap.add_argument('--outdir',default=None)
  args=ap.parse_args();planfile=Path(args.plan);base=planfile.parent;plan=json.loads(planfile.read_text())
  out=Path(args.outdir) if args.outdir else base/'summary';out.mkdir(parents=True,exist_ok=True)
  gold=pd.read_csv(args.test_manifest); goldidx=gold.Original_Row_Index.astype(int).tolist()
  records=[];missing=[]
  for rep in plan['repeats']:
    rp=base/'runs'/f'repeat_{rep["repeat"]:02d}_seed_{rep["base_seed"]}'
    results=rp/'baseline_metrics.csv'
    if not results.exists():missing.append(str(results));continue
    metrics=pd.read_csv(results)
    for key in plan['models']:
      pred=rp/key/'word_level_predictions.csv'
      if not pred.exists():raise FileNotFoundError(pred)
      actual=pd.read_csv(pred)
      if actual.Original_Row_Index.astype(int).tolist()!=goldidx:raise AssertionError(f'Test row index mismatch: {pred}')
      for col in ['Word_Tokens_Evaluated','Gold_BIO']:
        if not actual[col].astype(str).equals(gold[col].astype(str)):raise AssertionError(f'Gold data mismatch in {col}: {pred}')
      mapped={'mbert':'mBERT','indicbert':'IndicBERT','xlmr':'XLM-R'}[key]
      sub=metrics.loc[(metrics.Model==mapped)&(metrics.Metric=='micro avg')]
      if len(sub)!=1:raise AssertionError(f'No unique micro row for {mapped} in {results}')
      r=sub.iloc[0];assert int(r.Supervised_Train_Reviews)==64 and int(r.Supervised_Validation_Reviews)==16
      records.append({'repeat':rep['repeat'],'model':mapped,'seed':rep['model_seeds'][key],
                      'precision':r.Precision,'recall':r.Recall,'micro_f1':r.F1,
                      'support':r.Support,'prediction_sha256':hashlib.sha256(pred.read_bytes()).hexdigest()})
  if missing:
    print('INCOMPLETE: missing',len(missing),'pre-registered repeats:')
    for p in missing:print(' ',p)
    # Do not promote partial results as a final claim
    (out/'INCOMPLETE.txt').write_text('\n'.join(missing)+'\n')
    raise SystemExit('Report is incomplete; no final summary was produced.')
  allr=pd.DataFrame(records);assert len(allr)==15
  allr.to_csv(out/'all_seed_results.csv',index=False)
  grouped=allr.groupby('model').agg(mean_f1=('micro_f1','mean'),sd_f1=('micro_f1','std'),min_f1=('micro_f1','min'),
                                    max_f1=('micro_f1','max'),mean_recall=('recall','mean'),sd_recall=('recall','std'),n=('seed','count'))
  grouped['fixed_pipeline_f1_minus_mean_baseline_f1']=0.5088495575221239-grouped['mean_f1']
  grouped.to_csv(out/'seed_stability_summary.csv')
  print('\nALL runs:\n',allr.to_string(index=False))
  print('\nSummary (descriptive; same fixed test set across seeds):\n',grouped.to_string())
  print('Saved',out)
if __name__=='__main__':main()
