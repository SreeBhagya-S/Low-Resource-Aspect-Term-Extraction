"""Diagnostic: classify ORIGINAL lexicon literal forms by review-text exposure.

Uses parse_aspect_lexicon(), LexiconMatcher.match(), Normalizer(), surface_tokenize()
from the uploaded ORIGINAL ate_pipeline_core.py. Does not alter pipeline logic.

Important limitations:
 - Counts are a fresh, declared *lexical-candidate-exposure diagnostic*, NOT the
   original claimed Table 5 unless the old definition and corpus partition match.
 - Comparison excludes ALL text-matching evaluation source rows (duplicate-safe);
   hence its non-evaluation partition may differ from an earlier row-only analysis.
 - A leakage-free lexicon construction cannot be proven by exposure counts.
 - Recall ceilings must be regenerated with verified token/span alignment separately.

Example:
 py 04_diagnostic_lexicon_exposure.py --repo Low-Resource-Aspect-Term-Extraction-main --source "Balanced_CM (1).csv" --outdir R2_checks/exposure
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
import pandas as pd


def key(text):
    return ' '.join(unicodedata.normalize('NFKC',str(text)).casefold().split())


def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--outdir',default=Path('R2_checks/exposure'),type=Path)
    args=ap.parse_args()
    repo=args.repo.resolve(); source=args.source.resolve(); out=args.outdir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    # The import is deliberate: use ORIGINAL code, not a new matching heuristic.
    sys.path.insert(0,str(repo/'code'))
    from ate_pipeline_core import (
        parse_aspect_lexicon, LexiconMatcher, Normalizer, CORE_MAL_SUFFIXES,
        normalize_basic, load_union, load_word_set, clean_review_text, surface_tokenize,
    )
    resources=repo/'resources'
    lexpath=resources/'aspect_lexicon.txt'
    _,entries=parse_aspect_lexicon(str(lexpath))
    literal={e.normalized for e in entries if not e.regex and not e.category_trigger}
    # Same normalizer/matcher construction as original pipeline, excluding mBERT.
    suffixes=load_union([str(resources/'malayalam_suffix_1.txt'),str(resources/'malayalam_suffixmorph.txt')])
    suffixes |= {normalize_basic(s) for s in CORE_MAL_SUFFIXES}
    normalizer=Normalizer(suffixes)
    matcher=LexiconMatcher(entries)
    stopwords={normalize_basic(w) for w in load_word_set(str(resources/'merged_stopwords.txt'))}
    sentiment={normalize_basic(w) for w in load_union([str(resources/f) for f in [
        'negative-words_english.txt','neg_word_malayalam.txt','positive-words_english.txt','pos_words_malayalam.txt']])}
    source_df=pd.read_csv(source)
    eval_df=pd.read_csv(repo/'data/evaluation_input.csv')
    eval_text={key(t) for t in eval_df.Review_Text}
    # Preserve all 5,000 source rows and quarantine text-duplicates of eval reviews.
    is_eval_text=source_df.Review_Text.map(key).isin(eval_text)
    groups=[('evaluation',eval_df),('source_text_disjoint',source_df.loc[~is_eval_text])]
    found={g:set() for g,_ in groups}
    for label,df in groups:
        print('Scanning',label,'rows=',len(df),flush=True)
        for number,txt in enumerate(df.Review_Text,1):
            tokens=surface_tokenize(clean_review_text(txt))
            variants=[normalizer.variants(t) for t in tokens]
            candidates=matcher.match(tokens,variants,stopwords,sentiment,normalizer,0.75)
            found[label].update(c.canonical for c in candidates if c.canonical in literal)
            if number % 500 == 0: print('  ',number,flush=True)
    rows=[]
    for f in sorted(literal):
        ev=f in found['evaluation']; src=f in found['source_text_disjoint']
        status='both' if ev and src else ('evaluation_only' if ev else ('source_only' if src else 'neither'))
        rows.append({'normalized_literal':f,'exposed_in_evaluation':int(ev),
                     'exposed_in_text_disjoint_source':int(src),'exposure_class':status})
    pd.DataFrame(rows).to_csv(out/'current_matcher_literal_exposure.csv',index=False,encoding='utf-8-sig')
    summary={'lexicon_sha256':hashfile(lexpath),'source_sha256':hashfile(source),
             'full_source_rows':len(source_df),'evaluation_rows':len(eval_df),
             'text_disjoint_source_rows':int((~is_eval_text).sum()),
             'quarantined_source_rows':int(is_eval_text.sum()),
             'parser_distinct_literal_forms':len(literal),
             'distribution':pd.DataFrame(rows).exposure_class.value_counts().to_dict(),
             'method':'Original LexiconMatcher + original Normalizer; fuzzy cutoff 0.75; no mBERT/context',
             'warning':'Diagnostic is not proof of historical lexicon provenance or original Table 5 recall ceilings'}
    (out/'current_matcher_exposure_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('DO NOT copy these counts into Table 5 until you reconcile normalization and original definitions.')

if __name__=='__main__':main()
