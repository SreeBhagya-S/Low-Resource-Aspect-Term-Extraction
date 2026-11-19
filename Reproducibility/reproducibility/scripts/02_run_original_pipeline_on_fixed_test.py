"""Run unchanged repository ATE inference on the ORIGINAL fixed 200-review test set.

Precondition: run 01_prepare_fixed_protocol.py. No threshold changes, no data edits,
no lexicon edits. This reruns a pre-existing fixed system; if you have already used
this test to tune it, it must NOT be described as a new untouched confirmation.

Example:
 py 02_run_original_pipeline_on_fixed_test.py --repo Low-Resource-Aspect-Term-Extraction-main --prepared R2_checks --outdir R2_test_run
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import pandas as pd


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, logfile):
    print('RUN:',subprocess.list2cmdline(list(map(str,cmd))))
    with logfile.open('w',encoding='utf-8') as handle:
        returncode = subprocess.call(list(map(str,cmd)), stdout=handle, stderr=subprocess.STDOUT)
    if returncode:
        print('\n'.join(logfile.read_text(encoding='utf-8', errors='replace').splitlines()[-35:]))
        raise SystemExit(f'ERROR {returncode}: see {logfile}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--prepared',default=Path('R2_checks'),type=Path)
    ap.add_argument('--outdir',default=Path('R2_test_run'),type=Path)
    ap.add_argument('--device',choices=['cpu','cuda'],default=None)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    repo=a.repo.resolve(); prepared=a.prepared.resolve(); out=a.outdir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    input200=prepared/'test_input_200.csv'
    if not input200.is_file():
        raise SystemExit(f'Missing {input200}. Run 01_prepare_fixed_protocol.py first.')
    if len(pd.read_csv(input200)) != 200:
        raise SystemExit('Prepared test input is not 200 reviews')

    r=repo/'resources'; c=repo/'code'; d=repo/'data'; cfg=repo/'config/pipeline_config.json'
    pred=out/'original_pipeline_predictions_200.csv'
    cmd=[sys.executable,c/'generate_ate_bio.py','--config',cfg,
         '--input',input200,'--output',pred,
         '--lexicon',r/'aspect_lexicon.txt',
         '--stopwords',r/'merged_stopwords.txt']
    for f in ['malayalam_suffix_1.txt','malayalam_suffixmorph.txt']:
        cmd.extend(['--suffix-file',r/f])
    for f in ['negative-words_english.txt','neg_word_malayalam.txt',
              'positive-words_english.txt','pos_words_malayalam.txt']:
        cmd.extend(['--sentiment-file',r/f])
    if a.device: cmd.extend(['--device',a.device])
    cmd.extend(['--diagnostics',out/'candidate_diagnostics.csv'])

    evalcmd=[sys.executable,c/'evaluate_ate.py','--gold',d/'evaluation_gold.csv',
             '--pred',pred,'--manifest',d/'fixed_test_manifest_seed42.csv',
             '--output-dir',out/'metrics','--system-name','Original fixed ATE pipeline']
    meta={'python':sys.version,'platform':platform.platform(),
          'input_200_sha256':sha256(input200),
          'lexicon_sha256':sha256(r/'aspect_lexicon.txt'),
          'config_sha256':sha256(cfg),
          'original_pipeline_core_sha256':sha256(c/'ate_pipeline_core.py'),
          'generator_sha256':sha256(c/'generate_ate_bio.py'),
          'command':list(map(str,cmd)),
          'evaluation_command':list(map(str,evalcmd))}
    (out/'run_commands_and_hashes.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    if a.dry_run:
        print('DRY RUN: commands saved; original pipeline was not executed.'); return
    run(cmd,out/'inference.log')
    if len(pd.read_csv(pred))!=200:
        raise SystemExit('Prediction count differs from 200: no metrics reported.')
    run(evalcmd,out/'evaluation.log')
    print('Success: generated predictions, strict IOB2 metrics, and command/hash provenance in:',out)
    print('IMPORTANT: Compare the generated numbers against Tables 3 and 4; do not assume they match.')


if __name__=='__main__':
    main()
