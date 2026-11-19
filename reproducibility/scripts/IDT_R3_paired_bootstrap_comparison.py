#!/usr/bin/env python3
"""Strict IOB2 paired whole-review bootstrap for archived 200-review ATE predictions.
Usage: python IDT_R3_paired_bootstrap_comparison.py PATH_TO_CONSOLIDATED_ZIP
Records are aligned rowwise by fixed manifest; does not rerun models or change predictions.
"""
import sys,io,ast,json,hashlib
from zipfile import ZipFile
from pathlib import Path
import numpy as np
import pandas as pd

def spans(tags):
    result=[]; start=None; category=None
    for i,t in enumerate([*tags,'O']):
        if t=='O':
            if start is not None: result.append((start,i,category))
            start=None; category=None
        elif t.startswith('B-'):
            if start is not None:result.append((start,i,category))
            start=i;category=t[2:]
        elif t.startswith('I-'):
            if start is None or category != t[2:]:
                raise ValueError(f'Invalid strict IOB2 continuation: {t} at {i}')
        else:
            raise ValueError(f'Invalid tag: {t}')
    return set(result)

def micro(row_counts):
    tp,fp,fn=np.asarray(row_counts).sum(axis=-2).T
    return np.divide(2*tp,2*tp+fp+fn,where=2*tp+fp+fn>0)

def main(path):
    with ZipFile(path) as z:
        pref='IDT_R3_ALL_EVIDENCE_SINGLE_FOLDER/01_PUBLIC_REPRODUCIBILITY/selected_FINAL_2026_10_02/'
        items={
            'pipeline':pref+'proposed_pipeline_fixed200/PROPOSED_PIPELINE_FIXED200_PREDICTIONS.csv',
            'xlmr':pref+'low_resource/budget_0080/repeat_00/xlmr/word_level_predictions.csv',
            'manifest':pref+'fixed_test_manifest_seed42.csv',
        }
        raw={k:z.read(v) for k,v in items.items()}
    a=pd.read_csv(io.BytesIO(raw['pipeline']));b=pd.read_csv(io.BytesIO(raw['xlmr']));m=pd.read_csv(io.BytesIO(raw['manifest']))
    assert len(a)==len(b)==len(m)==200
    for col in ['Original_Row_Index','Gold_BIO','Word_Tokens_Evaluated']:
        assert a[col].astype(str).equals(b[col].astype(str)) and a[col].astype(str).equals(m[col].astype(str)),f'Mismatch: {col}'
    def counts(g,p):
        gold=spans(ast.literal_eval(g));pred=spans(ast.literal_eval(p))
        return len(gold & pred), len(pred-gold),len(gold-pred)
    pa=np.array([counts(row.Gold_BIO,row.Pipeline_BIO) for row in a.itertuples(index=False)],dtype=np.int64)
    pb=np.array([counts(row.Gold_BIO,row.Predicted_BIO) for row in b.itertuples(index=False)],dtype=np.int64)
    assert np.array_equal(pa[:,0]+pa[:,2],pb[:,0]+pb[:,2])
    def stat(c):
        t,f,n=c.sum(0);return {'tp':int(t),'fp':int(f),'fn':int(n),'micro_f1':float(2*t/(2*t+f+n)), 'precision':float(t/(t+f)),'recall':float(t/(t+n))}
    sa,sb=stat(pa),stat(pb)
    print('pipeline',sa,'\nxlmr',sb)
    assert (sa['tp'],sa['fp'],sa['fn'])==(115,195,27)
    assert round(sb['micro_f1'],3)==.470
    rng=np.random.default_rng(42)
    ids=rng.integers(0,200,size=(10000,200))
    a_s=pa[ids].sum(1);b_s=pb[ids].sum(1)
    fa=2*a_s[:,0]/(2*a_s[:,0]+a_s[:,1]+a_s[:,2])
    fb=2*b_s[:,0]/(2*b_s[:,0]+b_s[:,1]+b_s[:,2])
    dif=fa-fb
    ra=a_s[:,0]/(a_s[:,0]+a_s[:,2]);rb=b_s[:,0]/(b_s[:,0]+b_s[:,2])
    pa_s=a_s[:,0]/(a_s[:,0]+a_s[:,1]);pb_s=b_s[:,0]/(b_s[:,0]+b_s[:,1])
    # 2 sided percentile interval and directional distribution only (not significance under null)
    report={
        'purpose':'Exploratory paired whole-review bootstrap of difference in strict category-aware span micro-F1',
        'input_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in raw.items()},
        'test_review_count':200,'resamples':10000,'seed':42,'rng':'numpy.random.default_rng',
        'resampling_unit':'same sampled review indices applied to both models, with replacement',
        'comparison':'Original ATE pipeline minus low-resource XLM-R',
        'pipeline':sa,'xlmr':sb,
        'observed_micro_f1_difference':sa['micro_f1']-sb['micro_f1'],
        'paired_bootstrap_95_percentile_interval':list(map(float,np.quantile(dif,[.025,.975],method='linear'))),
        'fraction_bootstrap_differences_le_zero':float(np.mean(dif<=0)),
        'observed_recall_difference':sa['recall']-sb['recall'],
        'paired_bootstrap_recall_difference_95_percentile_interval':list(map(float,np.quantile(ra-rb,[.025,.975],method='linear'))),
        'observed_precision_difference':sa['precision']-sb['precision'],
        'paired_bootstrap_precision_difference_95_percentile_interval':list(map(float,np.quantile(pa_s-pb_s,[.025,.975],method='linear'))),
        'cautions':['A percentile bootstrap interval is not itself a valid randomization test p-value; do not present bootstrap sign fraction as a p-value.','Only one XLM-R training seed and one 80-review development pool are represented.','Historical lexicon independence is not established by these results.']
    }
    outfile=Path('/mnt/data/IDT_R3_paired_bootstrap_comparison.json')
    outfile.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    print('Saved',outfile)
if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else '/mnt/data/IDT_R3_ALL_EVIDENCE_SINGLE_FOLDER.zip')
