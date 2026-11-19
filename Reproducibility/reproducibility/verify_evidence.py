#!/usr/bin/env python3
"""Verify SHA-256 integrity and principal fixed-test evidence, without training."""
from pathlib import Path
import csv,hashlib,json,sys
root=Path(__file__).resolve().parent
with (root/'FILE_SHA256.csv').open(encoding='utf-8',newline='') as fh:
 rows=list(csv.DictReader(fh))
for item in rows:
 p=root/item['path']
 assert p.is_file(),f'Missing: {item["path"]}'
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==item['sha256'],f'Changed: {item["path"]}'
print(f'PASS: {len(rows)} file hashes')
def csvrow(path,when):
 with (root/path).open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if when(r):return r
 raise AssertionError(f'No matching row in {path}')
r=csvrow('results/pipeline/original_strict_span_metrics.csv',lambda x:x['Metric']=='micro avg')
assert abs(float(r['F1'])-0.5088495575221239)<1e-11
assert abs(float(r['Recall'])-115/142)<1e-11
r2=csvrow('results/pipeline/pruned17_strict_span_metrics.csv',lambda x:x['Metric']=='micro avg')
assert abs(float(r2['F1'])-0.5111111111111111)<1e-11
assert abs(float(r2['Recall'])-115/142)<1e-11
with (root/'results/lexicon/exact_fuzzy_exposure_summary.json').open(encoding='utf-8') as fh: x=json.load(fh)
assert x['operational_exact']=={'both':239,'source_only':102,'evaluation_only':17,'neither':159}
assert sum(x['operational_exact'].values())==517
with (root/'results/lexicon/gold_candidate_coverage_summary.json').open(encoding='utf-8') as fh:y=json.load(fh)
assert y['gold_spans_in_aligned_rows']==791
assert y['raw_lexical_candidate_pre_context']['covered_gold_spans']==755
with (root/'results/statistics/paired_bootstrap_comparison.json').open(encoding='utf-8') as fh:z=json.load(fh)
assert (z['pipeline']['tp'],z['pipeline']['fp'],z['pipeline']['fn'])==(115,195,27)
assert (z['xlmr']['tp'],z['xlmr']['fp'],z['xlmr']['fn'])==(67,76,75)
for regime,expected in [('low_resource',{'mbert':.352,'xlmr':.470,'indicbert':.125,'bilstm':.006}),('full_supervision',{'mbert':.746,'xlmr':.734,'indicbert':.711,'bilstm':.617})]:
 for model,target in expected.items():
  line=csvrow(f'results/baselines/{regime}/{model}/metrics.csv',lambda r:r['Metric']=='micro avg')
  assert abs(float(line['F1'])-target)<0.0006,(regime,model,line['F1'])
with (root/'results/ablation/component_ablation.csv').open(encoding='utf-8-sig',newline='') as f:
 a=list(csv.DictReader(f))
assert len(a)==7 and a[0]['Configuration']=='Full pipeline'
assert abs(float(a[4]['F1'])-0.4298093587521663)<1e-10
with (root/'results/statistics/seed_stability_summary.csv').open(encoding='utf-8-sig',newline='') as f:
 rows=list(csv.DictReader(f))
assert len(rows)==3 and all(int(v['n'])==5 for v in rows)
print('PASS: pipeline, lexicon, candidate coverage, baselines, ablations and seed summary')
