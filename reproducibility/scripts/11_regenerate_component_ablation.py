#!/usr/bin/env python3
"""Execute the seven pre-recorded fixed-test ablation configurations against the existing ATE code.

Requires the repository's code/generate_ate_bio.py and resources, a local
bert-base-multilingual-cased model, and the exact original fixed 200-row input.
The archived result CSVs are not modified; new predictions are written to a separate directory.
"""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--repo',type=Path,default=Path('.'))
p.add_argument('--outdir',type=Path,required=True)
p.add_argument('--device',default=None)
a=p.parse_args()
repo=a.repo.resolve()
repro=repo/'reproducibility'
run=repo/'code'/'generate_ate_bio.py'
if not run.is_file(): raise FileNotFoundError(run)
test=repro/'protocol'/'test_input_200.csv'
if not test.is_file():raise FileNotFoundError(test)
out=a.outdir.resolve();out.mkdir(parents=True,exist_ok=True)
for i in range(7):
 cmd=[sys.executable,str(run),
  '--config',str(repro/'results'/'ablation'/f'config_{i:02d}.json'),
  '--input',str(test),
  '--output',str(out/f'prediction_{i:02d}.csv'),
  '--lexicon',str(repo/'resources'/'aspect_lexicon.txt'),
  '--suffix-file',str(repo/'resources'/'malayalam_suffix_1.txt'),
  '--suffix-file',str(repo/'resources'/'malayalam_suffixmorph.txt'),
  '--stopwords',str(repo/'resources'/'merged_stopwords.txt'),
  '--sentiment-file',str(repo/'resources'/'neg_word_malayalam.txt'),
  '--sentiment-file',str(repo/'resources'/'negative-words_english.txt'),
  '--sentiment-file',str(repo/'resources'/'pos_words_malayalam.txt'),
  '--sentiment-file',str(repo/'resources'/'positive-words_english.txt')]
 if a.device:cmd.extend(['--device',a.device])
 subprocess.run(cmd,check=True)
 print(f'Configuration {i}/6 complete')
