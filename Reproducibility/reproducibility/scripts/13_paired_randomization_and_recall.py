#!/usr/bin/env python3
"""Strict IOB2, category-aware paired randomization and whole-review bootstrap.

Inputs are saved model predictions, not scores reconstructed from a paper table.
The paired randomization null exchanges the two systems' predicted tag sequences
within each review; the bootstrap samples whole reviews with replacement.

Usage: python 13_paired_randomization_and_recall.py --data-dir data --outdir results
"""
import argparse
import ast
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_tags(v):
    seq = v if isinstance(v, list) else ast.literal_eval(str(v))
    if not isinstance(seq, list):
        raise ValueError('BIO sequence must be a list')
    return [str(x) for x in seq]


def spans(tags):
    found=[]; start=None; category=None
    for i, tag in enumerate([*tags,'O']):
        if tag == 'O':
            if start is not None: found.append((start,i,category))
            start=None;category=None
        elif tag.startswith('B-'):
            if start is not None: found.append((start,i,category))
            start=i;category=tag[2:]
            if not category: raise ValueError('Empty category')
        elif tag.startswith('I-'):
            if start is None or category != tag[2:]:
                raise ValueError(f'Invalid strict IOB2 transition: {tag} at index {i}')
        else:
            raise ValueError(f'Invalid tag {tag!r}')
    return set(found)


def row_count(gold, prediction):
    g,p=spans(gold),spans(prediction)
    return (len(g&p),len(p-g),len(g-p))


def metric(sum_counts):
    t,f,n = map(float, sum_counts)
    return {
        'TP': int(t), 'FP': int(f), 'FN': int(n),
        'precision': t/(t+f) if t+f else 0.0,
        'recall': t/(t+n) if t+n else 0.0,
        'micro_f1': 2*t/(2*t+f+n) if 2*t+f+n else 0.0,
    }


def batch_scores(c):
    t,f,n = c[:,0],c[:,1],c[:,2]
    f1=np.divide(2*t,2*t+f+n,out=np.zeros_like(t,dtype=float),where=(2*t+f+n)!=0)
    recall=np.divide(t,t+n,out=np.zeros_like(t,dtype=float),where=(t+n)!=0)
    return f1,recall


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',default='data')
    ap.add_argument('--outdir',default='results')
    ap.add_argument('--randomizations',type=int,default=100000)
    ap.add_argument('--bootstraps',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if args.randomizations<1 or args.bootstraps<1:raise ValueError('Sample counts must be positive')
    datadir=Path(args.data_dir);outdir=Path(args.outdir);outdir.mkdir(parents=True,exist_ok=True)
    files={
      'pipeline':datadir/'PROPOSED_PIPELINE_FIXED200_PREDICTIONS.csv',
      'xlmr':datadir/'XLMR_LOW_RESOURCE_FIXED200_PREDICTIONS.csv',
      'manifest':datadir/'fixed_test_manifest_seed42.csv',
    }
    for label,p in files.items():
        if not p.is_file(): raise FileNotFoundError(f'Missing {label}: {p}')
    a=pd.read_csv(files['pipeline']);b=pd.read_csv(files['xlmr']);m=pd.read_csv(files['manifest'])
    if not (len(a)==len(b)==len(m)==200):raise AssertionError('Expected aligned 200-row fixed test predictions')
    for column in ('Original_Row_Index','Word_Tokens_Evaluated','Gold_BIO'):
        if not (a[column].astype(str).equals(b[column].astype(str)) and a[column].astype(str).equals(m[column].astype(str))):
            raise AssertionError(f'Mismatch in fixed test alignment: {column}')
    counts_a=[];counts_b=[]
    for i in range(len(m)):
        gold=parse_tags(a.iloc[i]['Gold_BIO'])
        aa=parse_tags(a.iloc[i]['Pipeline_BIO'])
        bb=parse_tags(b.iloc[i]['Predicted_BIO'])
        words=parse_tags(a.iloc[i]['Word_Tokens_Evaluated'])
        if len(gold)!=len(aa) or len(gold)!=len(bb) or len(gold)!=len(words):raise AssertionError(f'Tag lengths differ at review {i}')
        counts_a.append(row_count(gold,aa));counts_b.append(row_count(gold,bb))
    pa=np.asarray(counts_a,dtype=np.int64);pb=np.asarray(counts_b,dtype=np.int64)
    if not np.array_equal(pa[:,0]+pa[:,2],pb[:,0]+pb[:,2]):raise AssertionError('Gold span support mismatch')
    sa=metric(pa.sum(0));sb=metric(pb.sum(0))
    assert (sa['TP'],sa['FP'],sa['FN'])==(115,195,27),(sa,sb)
    assert (sb['TP'],sb['FP'],sb['FN'])==(67,76,75),(sa,sb)
    observed_f1=sa['micro_f1']-sb['micro_f1']
    observed_recall=sa['recall']-sb['recall']
    rng=np.random.default_rng(args.seed)
    ext_f1=0;pos_f1=0;ext_recall=0;pos_recall=0;done=0
    # Exchange predictions within each *review* (not independently per aspect span).
    # Include the unmodified assignment as a valid Monte Carlo null draw.
    while done<args.randomizations:
        batch=min(5000,args.randomizations-done)
        swap=rng.integers(0,2,size=(batch,len(pa)),dtype=np.int8).astype(bool)
        pfirst=np.where(swap[:,:,None],pb[None,:,:],pa[None,:,:]).sum(axis=1)
        psecond=np.where(swap[:,:,None],pa[None,:,:],pb[None,:,:]).sum(axis=1)
        fx,rx=batch_scores(pfirst);fy,ry=batch_scores(psecond)
        d1=fx-fy;d2=rx-ry
        ext_f1+=int(np.count_nonzero(np.abs(d1)>=abs(observed_f1)-1e-12))
        pos_f1+=int(np.count_nonzero(d1>=observed_f1-1e-12))
        ext_recall+=int(np.count_nonzero(np.abs(d2)>=abs(observed_recall)-1e-12))
        pos_recall+=int(np.count_nonzero(d2>=observed_recall-1e-12))
        done+=batch
    # add-one p-value avoids claiming zero p under finite Monte Carlo sampling
    perm={
      'f1_two_sided_p':(ext_f1+1)/(args.randomizations+1),
      'f1_one_sided_pipeline_greater_p':(pos_f1+1)/(args.randomizations+1),
      'recall_two_sided_p':(ext_recall+1)/(args.randomizations+1),
      'recall_one_sided_pipeline_greater_p':(pos_recall+1)/(args.randomizations+1),
    }
    # Exact paired sign-flip test is feasible for recall because the gold-span
    # denominator is shared, and recall differences reduce to integer TP differences.
    nonzero=np.abs((pa[:,0]-pb[:,0]).astype(int))
    nonzero=nonzero[nonzero>0]
    pmf={0:1}
    for magnitude in nonzero:
        new=defaultdict(int)
        for sum_tp,ways in pmf.items():
            new[sum_tp+int(magnitude)]+=ways
            new[sum_tp-int(magnitude)]+=ways
        pmf=new
    observed_tp_diff=abs(int(pa[:,0].sum()-pb[:,0].sum()))
    exact_recall_extreme=sum(ways for sum_tp,ways in pmf.items() if abs(sum_tp)>=observed_tp_diff)
    exact_recall_p=exact_recall_extreme/(2**len(nonzero))
    # Separate, fixed bootstrap random stream matches the independent 10k review
    # bootstrap audit, rather than consuming the permutation RNG state.
    bootstrap_rng=np.random.default_rng(args.seed)
    indices=bootstrap_rng.integers(0,len(pa),size=(args.bootstraps,len(pa)))
    ab=pa[indices].sum(axis=1);bb=pb[indices].sum(axis=1)
    f1a,ra=batch_scores(ab);f1b,rb=batch_scores(bb)
    interval=lambda ar:[float(x) for x in np.quantile(ar,[.025,.975],method='linear')]
    report={
      'method':'strict category-aware IOB2 exact-span metrics; whole-review paired exchange randomization under label-exchangeability null',
      'disclaimer':'Single fixed-trained XLM-R model versus fixed pipeline. Randomization p-values test paired output-exchangeability conditional on these systems and reviews; they do not assess variation across XLM-R training seeds or establish external validity.',
      'files_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()},
      'test_reviews':len(pa),'gold_spans':int(pa[:,0].sum()+pa[:,2].sum()),
      'pipeline':sa,'xlmr':sb,
      'observed':{'pipeline_minus_xlmr_micro_f1':observed_f1,'pipeline_minus_xlmr_recall':observed_recall},
      'paired_randomization':{'randomizations':args.randomizations,'seed':args.seed,
                              'randomization_unit':'whole review, swap model prediction vectors with probability 1/2',
                              'two_sided_test':'absolute difference >= observed absolute difference',
                              'monte_carlo_p_correction':'(extreme_count+1)/(randomizations+1)',
                              'recall_exact_two_sided_signflip_p':exact_recall_p,
                              'recall_exact_nonzero_review_contributions':int(len(nonzero)),
                              'recall_exact_null_total_assignments':str(2**len(nonzero)),
                              **perm},
      'paired_bootstrap':{'resamples':args.bootstraps,'rng':'default_rng','seed':args.seed,
                          'note':'Independent default_rng initialized with same seed as separate historical bootstrap audit',
                          'micro_f1_difference_95pct_percentile_CI':interval(f1a-f1b),
                          'recall_difference_95pct_percentile_CI':interval(ra-rb)},
      'limitations':['Exploratory/post-hoc; test set has been examined during revisions',
                     'Permutation null assumes paired model outputs can be exchanged at review level',
                     'No multiple-comparison adjustment',
                     'Does not replace repeated training seeds or a newly annotated independent test'],
    }
    out=outdir/'paired_randomization_recall_results.json'
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Pipeline F1={sa["micro_f1"]:.6f}, recall={sa["recall"]:.6f}; XLM-R F1={sb["micro_f1"]:.6f}, recall={sb["recall"]:.6f}')
    print(f'Observed differences: F1={observed_f1:+.6f}, recall={observed_recall:+.6f}')
    for k,v in perm.items():print(f'{k}={v:.7f}')
    print(f'Exact paired recall sign-flip p={exact_recall_p:.12g} (two-sided)')
    print(f'Paired bootstrap F1 95% CI: {report["paired_bootstrap"]["micro_f1_difference_95pct_percentile_CI"]}')
    print(f'Paired bootstrap recall 95% CI: {report["paired_bootstrap"]["recall_difference_95pct_percentile_CI"]}')
    print('Saved',out)
if __name__=='__main__':main()
