r"""Read-only Reviewer 2 / Table 5 audit: ORIGINAL matcher exact vs fuzzy exposure.

From PowerShell in IDT_R3\R3:
  py 06_compare_exact_fuzzy_exposure.py --repo "..\GitHub_R2" --source ".\Balanced_CM (1).csv" --outdir "R2_checks\exact_fuzzy"

This loads the original parser, normalizer, and matcher, and does NOT edit any
model/configuration/source data. Its 'operational exact' classification includes
normalizer variants (lemmatization/suffixes), but excludes fuzzy and regex hits.
It is not a test of historical lexicon independence nor of gold-span recall.
"""
import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd


def text_key(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def hash_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            digest.update(b)
    return digest.hexdigest()


def status(eval_hit, source_hit):
    if eval_hit and source_hit:
        return 'both'
    if eval_hit:
        return 'evaluation_only'
    if source_hit:
        return 'source_only'
    return 'neither'


def distribution(terms, ev, src):
    count = Counter(status(t in ev, t in src) for t in terms)
    return {k: count[k] for k in ('both', 'source_only', 'evaluation_only', 'neither')}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, default=Path('R2_checks/exact_fuzzy'))
    ap.add_argument('--limit', type=int, default=None, help='For smoke testing only; NEVER use limited outputs in manuscript.')
    args = ap.parse_args()
    root = args.repo.resolve()
    source = args.source.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    lexicon = root / 'resources' / 'aspect_lexicon.txt'
    if not lexicon.is_file() or not source.is_file():
        raise FileNotFoundError(f'Missing lexicon {lexicon} or source {source}')

    # Import and execute the ORIGINAL matching algorithm without modifications.
    sys.path.insert(0, str(root / 'code'))
    from ate_pipeline_core import (
        CORE_MAL_SUFFIXES, LexiconMatcher, Normalizer,
        clean_review_text, load_union, load_word_set,
        normalize_basic, parse_aspect_lexicon, surface_tokenize,
    )
    _, entries = parse_aspect_lexicon(str(lexicon))
    literals = sorted({e.normalized for e in entries if not e.regex and not e.category_trigger})
    literal_set = set(literals)
    matcher = LexiconMatcher(entries)
    resources = root / 'resources'
    suffixes = load_union([str(resources / x) for x in ('malayalam_suffix_1.txt', 'malayalam_suffixmorph.txt')])
    suffixes |= {normalize_basic(s) for s in CORE_MAL_SUFFIXES}
    normalizer = Normalizer(suffixes)
    stopwords = {normalize_basic(w) for w in load_word_set(str(resources / 'merged_stopwords.txt'))}
    sentiment_files = (
        'negative-words_english.txt', 'neg_word_malayalam.txt',
        'positive-words_english.txt', 'pos_words_malayalam.txt'
    )
    sentiment = {normalize_basic(w) for w in load_union([str(resources / f) for f in sentiment_files])}

    df_source = pd.read_csv(source)
    df_eval = pd.read_csv(root / 'data' / 'evaluation_input.csv')
    if 'Review_Text' not in df_source or 'Review_Text' not in df_eval:
        raise ValueError('Expected Review_Text column in both CSV inputs.')
    eval_texts = {text_key(t) for t in df_eval.Review_Text}
    quarantine = df_source.Review_Text.map(text_key).isin(eval_texts)
    groups = [('evaluation', df_eval), ('source_text_disjoint', df_source.loc[~quarantine])]
    if args.limit:
        groups = [(name, df.head(args.limit)) for name, df in groups]
        print('SMOKE-TEST ONLY: --limit specified; results must NOT be cited in manuscript.', flush=True)

    seen = {group: {'exact': set(), 'fuzzy': set(), 'exact_or_fuzzy': set(), 'script04_compatible': set()} for group, _ in groups}
    counters = {group: Counter() for group, _ in groups}
    for group, df in groups:
        print(f'Scanning {group}: {len(df)} rows', flush=True)
        for i, review in enumerate(df.Review_Text, start=1):
            tokens = surface_tokenize(clean_review_text(review))
            variants = [normalizer.variants(t) for t in tokens]
            candidates = matcher.match(tokens, variants, stopwords, sentiment, normalizer, 0.75)
            for candidate in candidates:
                # We audit only genuine lexicon literal entry candidates.
                # Category-heading triggers and regex patterns are counted separately.
                counters[group][candidate.match_type] += 1
                if candidate.canonical in literal_set:
                    seen[group]['script04_compatible'].add(candidate.canonical)
                if candidate.category_trigger or candidate.canonical not in literal_set:
                    continue
                if candidate.match_type == 'exact':
                    seen[group]['exact'].add(candidate.canonical)
                    seen[group]['exact_or_fuzzy'].add(candidate.canonical)
                elif candidate.match_type == 'fuzzy':
                    seen[group]['fuzzy'].add(candidate.canonical)
                    seen[group]['exact_or_fuzzy'].add(candidate.canonical)
            if i % 500 == 0:
                print(f'  {i}', flush=True)

    ev = seen['evaluation']
    src = seen['source_text_disjoint']
    rows = []
    for term in literals:
        rows.append({
            'normalized_literal': term,
            'operational_exact_class': status(term in ev['exact'], term in src['exact']),
            'fuzzy_only_added_class': status(term in ev['fuzzy']-ev['exact'], term in src['fuzzy']-src['exact']),
            'exact_plus_fuzzy_class': status(term in ev['exact_or_fuzzy'], term in src['exact_or_fuzzy']),
            'script04_compatible_class': status(term in ev['script04_compatible'], term in src['script04_compatible']),
            'exact_eval': int(term in ev['exact']),
            'exact_source': int(term in src['exact']),
            'fuzzy_eval': int(term in ev['fuzzy']),
            'fuzzy_source': int(term in src['fuzzy']),
        })
    pd.DataFrame(rows).to_csv(outdir / 'exact_vs_fuzzy_form_exposure.csv', index=False, encoding='utf-8-sig')
    results = {
        'note': 'This is a diagnostic of lexical candidate exposure, NOT a gold-span recall ceiling or a provenance audit.',
        'lexicon_sha256': hash_file(lexicon),
        'source_sha256': hash_file(source),
        'source_total': len(df_source),
        'evaluation_total': len(df_eval),
        'quarantined_source_text_matches': int(quarantine.sum()),
        'text_disjoint_source_total': int((~quarantine).sum()),
        'distinct_normalized_literal_forms': len(literals),
        'operational_exact': distribution(literals, ev['exact'], src['exact']),
        'exact_plus_fuzzy_excluding_heading_triggers': distribution(literals, ev['exact_or_fuzzy'], src['exact_or_fuzzy']),
        'script04_compatible_all_match_types': distribution(literals, ev['script04_compatible'], src['script04_compatible']),
        'candidate_match_types': {k: dict(v) for k,v in counters.items()},
        'cautions': [
            'Operational exact uses the original normalization variants and is not raw-text substring matching.',
            'Fuzzy matching at 0.75 is an independent candidate path in the original matcher.',
            'Category-heading triggers and regex patterns are excluded from the operational exact and exact-plus-fuzzy counts.',
            'The script04-compatible counts include all matcher candidate types when canonical text coincides with a literal, to cross-check the earlier 04 diagnostic.',
            'These counts do not prove historical lexicon independence or reproduce Table 5 recall ceilings.',
            'Results from a --limit smoke test cannot be used in the manuscript.'
        ],
    }
    (outdir / 'exact_fuzzy_exposure_summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'Saved {outdir / "exact_fuzzy_exposure_summary.json"}')
    print(f'Saved {outdir / "exact_vs_fuzzy_form_exposure.csv"}')
    print('Original pipeline and lexicon have not been modified.')

if __name__ == '__main__':
    main()
