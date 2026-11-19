#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Read-only R3 audit: for each gold BIO span, do active lexical candidates
cover its exact token boundary under the SAME category or another category?

PowerShell (from R3, with Script 08 in the same directory):

    py 09_audit_same_span_category_disagreement.py `
      --repo "..\GitHub_R2" `
      --outdir "R2_checks\category_disagreement"

Relies on Script 08 solely for its previously verified gold BIO parsing helpers.
Imports and uses the UNMODIFIED repo's tokenizer, normalizer and lexicon matcher.
Does not use contextual/mBERT inference; does not change lexicon or gold labels.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--outdir', type=Path, default=Path('R2_checks/category_disagreement'))
    ap.add_argument('--limit', type=int, default=None, help='For smoke test only; never cite partial outputs')
    args = ap.parse_args()
    repo = args.repo.resolve()
    outdir = args.outdir.resolve()

    helper = Path(__file__).resolve().with_name('08_audit_gold_span_candidate_coverage.py')
    if not helper.is_file():
        raise FileNotFoundError('Keep 08_audit_gold_span_candidate_coverage.py beside this script: ' + str(helper))
    audit08 = runpy.run_path(str(helper), run_name='audit08_helpers')
    as_list = audit08['as_list']
    get_tags = audit08['get_tags']
    strict_gold_spans = audit08['strict_gold_spans']
    hash_file = audit08['hash_file']

    lexicon_path = repo / 'resources' / 'aspect_lexicon.txt'
    gold_path = repo / 'data' / 'evaluation_gold.csv'
    config_path = repo / 'config' / 'pipeline_config.json'
    for required in (lexicon_path, gold_path, config_path, repo / 'code' / 'ate_pipeline_core.py'):
        if not required.is_file():
            raise FileNotFoundError(f'Missing required repository file: {required}')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    components = cfg.get('components', {})
    fuzzy_on = bool(components.get('fuzzy_matching', True))
    regex_on = bool(components.get('regex_matching', True))
    morphology_on = bool(components.get('morphology', True))
    guard_on = bool(components.get('category_guard', True))
    sentiment_filter_on = bool(components.get('stopword_sentiment_filter', True))
    taxonomy = cfg.get('taxonomy', 'evaluation16')

    sys.path.insert(0, str(repo / 'code'))
    from ate_pipeline_core import (  # repo originals; do not edit
        CORE_MAL_SUFFIXES, EVALUATION_CATEGORIES, BasicOnlyNormalizer,
        LexiconMatcher, Normalizer, category_specific_guard,
        clean_review_text, load_union, load_word_set, normalize_basic,
        parse_aspect_lexicon, surface_tokenize,
    )

    _, entries = parse_aspect_lexicon(str(lexicon_path))
    if taxonomy == 'evaluation16':
        entries = [e for e in entries if e.category in EVALUATION_CATEGORIES]
    elif taxonomy != 'lexicon':
        raise ValueError(f'Unknown taxonomy: {taxonomy!r}')
    matcher = LexiconMatcher(entries)
    resources = repo / 'resources'
    suffixes = load_union([str(resources / f) for f in
                           ('malayalam_suffix_1.txt', 'malayalam_suffixmorph.txt')])
    suffixes |= {normalize_basic(s) for s in CORE_MAL_SUFFIXES}
    base_normalizer = Normalizer(suffixes)
    if morphology_on and not base_normalizer.wordnet_available:
        raise RuntimeError('NLTK WordNet data unavailable; run: py -m nltk.downloader wordnet omw-1.4')
    normalizer = base_normalizer if morphology_on else BasicOnlyNormalizer()

    stopwords = {normalize_basic(w) for w in load_word_set(str(resources / 'merged_stopwords.txt'))}
    sentiment_files = (
        'negative-words_english.txt', 'neg_word_malayalam.txt',
        'positive-words_english.txt', 'pos_words_malayalam.txt',
    )
    sentiment = {normalize_basic(w) for w in load_union([str(resources / f) for f in sentiment_files])}
    if not sentiment_filter_on:
        stopwords, sentiment = set(), set()

    gold = pd.read_csv(gold_path)
    if len(gold) != 1000:
        raise ValueError(f'Expected 1000 annotated rows, got {len(gold)}')
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError('--limit must be positive')
        gold = gold.head(args.limit)
        print('SMOKE TEST ONLY: partial outputs must not be cited', flush=True)

    rows = []
    mismatch_rows = []
    category_matrix = Counter()
    strict = 0
    boundary_agnostic = 0
    wrong_category_only = 0
    no_exact_boundary = 0
    unfiltered_count = 0
    for i, record in gold.iterrows():
        tokens = [str(t) for t in as_list(record['Word_Tokens'], i, 'Word_Tokens')]
        tags = get_tags(record['BIO_Tags'], i)
        if len(tokens) != len(tags):
            raise ValueError(f'Token/tag count mismatch at row {i}')
        pipeline_tokens = surface_tokenize(clean_review_text(record['Review_Text']))
        if tokens != pipeline_tokens:
            mismatch_rows.append(int(i))
            continue
        spans = strict_gold_spans(tags, i)
        variants = [normalizer.variants(token) for token in tokens]
        raw = matcher.match(tokens, variants, stopwords, sentiment,
                            normalizer, float(cfg.get('fuzzy_cutoff', 0.75)))
        active = [c for c in raw if (fuzzy_on or c.match_type != 'fuzzy')
                  and (regex_on or c.match_type != 'regex')]
        guarded = ([c for c in active if category_specific_guard(c, tokens)]
                   if guard_on else active)
        by_boundary = defaultdict(list)
        by_boundary_guarded = defaultdict(list)
        for c in active:
            by_boundary[(c.start, c.end)].append(c)
        for c in guarded:
            by_boundary_guarded[(c.start, c.end)].append(c)
        for start, end, cat in spans:
            options = by_boundary.get((start, end), [])
            same_cat = [c for c in options if c.category == cat]
            guarded_options = by_boundary_guarded.get((start, end), [])
            unfiltered_correct = [c for c in guarded_options if c.category == cat and not c.filtered]
            option_categories = sorted({c.category for c in options})
            if same_cat:
                status = 'strict_correct_category'
                strict += 1
            elif options:
                status = 'same_boundary_wrong_category_only'
                wrong_category_only += 1
                for other_cat in option_categories:
                    category_matrix[(cat, other_cat)] += 1
            else:
                status = 'no_same_boundary_candidate'
                no_exact_boundary += 1
            boundary_agnostic += bool(options)
            unfiltered_count += bool(unfiltered_correct)
            context_left = max(0, start - 4)
            context_right = min(len(tokens), end + 4)
            row = {
                'original_row_index': int(i),
                'gold_start': int(start), 'gold_end': int(end),
                'gold_category': cat,
                'gold_surface': ' '.join(tokens[start:end]),
                'status': status,
                'candidate_categories_same_boundary': ';'.join(option_categories),
                'candidate_match_types_same_boundary': ';'.join(sorted({c.match_type for c in options})),
                'correct_category_before_context': int(bool(same_cat)),
                'any_category_same_boundary': int(bool(options)),
                'correct_category_unfiltered_after_guard': int(bool(unfiltered_correct)),
                'local_token_context': ' '.join(tokens[context_left:context_right]),
            }
            rows.append(row)
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(gold)}; gold spans={len(rows)}', flush=True)

    total = len(rows)
    if mismatch_rows:
        raise RuntimeError(f'Token mismatch on {len(mismatch_rows)} rows; aborting: {mismatch_rows[:20]}')
    if args.limit is None:
        if total != 791:
            raise AssertionError(f'Gold span total changed: expected 791, got {total}')
        if strict != 755:
            raise AssertionError(f'Strict coverage changed from Script 08: expected 755, got {strict}. Compare environments/configuration.')
        if unfiltered_count != 695:
            raise AssertionError(f'Unfiltered coverage changed from Script 08: expected 695, got {unfiltered_count}.')
        if wrong_category_only + no_exact_boundary != 36:
            raise AssertionError('Uncovered counts differ from Script 08')

    summary = {
        'scope': '1000 gold reviews; 791 BIO gold spans; original lexical matcher, active config',
        'smoke_test_only': args.limit is not None,
        'gold_spans': total,
        'strict_correct_category_and_boundary': {'count': strict, 'rate': round(strict / total, 8) if total else None},
        'category_blind_same_boundary': {'count': boundary_agnostic, 'rate': round(boundary_agnostic / total, 8) if total else None},
        'among_strict_uncovered_wrong_category_only': wrong_category_only,
        'among_strict_uncovered_no_same_boundary_candidate': no_exact_boundary,
        'correct_category_after_guard_unfiltered': unfiltered_count,
        'wrong_category_pairs': [
            {'gold_category': a, 'candidate_category': b, 'count': n}
            for (a, b), n in sorted(category_matrix.items(), key=lambda x: (-x[1], x[0]))
        ],
        'current_config': {'regex_matching': regex_on, 'fuzzy_matching': fuzzy_on,
                           'fuzzy_cutoff': cfg.get('fuzzy_cutoff', 0.75),
                           'morphology': morphology_on, 'category_guard': guard_on,
                           'stopword_sentiment_filter': sentiment_filter_on, 'taxonomy': taxonomy},
        'sha256': {'gold': hash_file(gold_path), 'lexicon': hash_file(lexicon_path),
                   'config': hash_file(config_path)},
        'caveats': [
            'Category-blind boundary recall is DIAGNOSTIC ONLY; it is not strict ATE recall and must not replace it.',
            'Gold annotations must not be altered based on model outputs.',
            'An alternative candidate category does not establish that the gold category is wrong.',
            'These counts do not reproduce the historical Table 5 pruned-lexicon experiment.',
        ],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / 'all_gold_spans_with_candidate_categories.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([r for r in rows if r['status'] != 'strict_correct_category']).to_csv(
        outdir / 'strict_uncovered_classified.csv', index=False, encoding='utf-8-sig')
    (outdir / 'category_disagreement_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('Saved:', outdir / 'category_disagreement_summary.json')
    print('Saved:', outdir / 'strict_uncovered_classified.csv')
    print('Saved:', outdir / 'all_gold_spans_with_candidate_categories.csv')
    print('Read-only diagnostic: no change to gold, lexicon or pipeline logic.', flush=True)


if __name__ == '__main__':
    main()
