#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Read-only Reviewer 2 / Table 5 audit of GOLD-span lexical candidate coverage.

Use from PowerShell in C:\Users\sreeb\Desktop\IDT_R3\R3:

    py 08_audit_gold_span_candidate_coverage.py `
      --repo "..\GitHub_R2" `
      --outdir "R2_checks\gold_candidate_coverage"

Uses the original repo's lexicon parser, tokenizer, normalization and matcher.
No pretrained model is loaded and no algorithm/resources are modified.

This reports the maximum number of gold spans exposed by raw lexical candidates
BEFORE mBERT contextual filtering and overlap selection. It is NOT the final
ATE system recall, not a lexicon-provenance audit and not a reproduction of
historical Table 5 pruning metrics.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def as_list(cell, row_id, column):
    try:
        result = ast.literal_eval(str(cell))
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Row {row_id}: cannot parse {column}: {exc}") from exc
    if not isinstance(result, list):
        raise ValueError(f"Row {row_id}: {column} is not a list")
    return result


def get_tags(cell, row_id):
    result = as_list(cell, row_id, 'BIO_Tags')
    if result and isinstance(result[0], (list, tuple)):
        return [str(p[1]) for p in result]
    return [str(t) for t in result]


def strict_gold_spans(tags, row_id):
    """Extract strict IOB2 gold spans as (start, end-exclusive, category)."""
    spans = []
    open_start = None
    open_cat = None
    for j, tag in enumerate([*tags, 'O']):
        if tag == 'O':
            if open_start is not None:
                spans.append((open_start, j, open_cat))
                open_start, open_cat = None, None
        elif tag.startswith('B-'):
            if open_start is not None:
                spans.append((open_start, j, open_cat))
            open_start, open_cat = j, tag[2:]
        elif tag.startswith('I-'):
            cat = tag[2:]
            if open_start is None or open_cat != cat:
                raise ValueError(f"Row {row_id}, token {j}: invalid IOB2 continuation {tag}")
        else:
            raise ValueError(f"Row {row_id}, token {j}: invalid BIO tag {tag!r}")
    return spans


def summarize_span_count(rows, key, total):
    recovered = sum(bool(row[key]) for row in rows)
    return {'covered_gold_spans': int(recovered), 'gold_spans': int(total),
            'coverage': round(recovered / total, 8) if total else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--outdir', default=Path('R2_checks/gold_candidate_coverage'), type=Path)
    p.add_argument('--limit', type=int, default=None,
                   help='Smoke-test only. NEVER quote --limit outputs in manuscript.')
    args = p.parse_args()
    repo, outdir = args.repo.resolve(), args.outdir.resolve()
    gold_path = repo / 'data' / 'evaluation_gold.csv'
    lexicon_path = repo / 'resources' / 'aspect_lexicon.txt'
    config_path = repo / 'config' / 'pipeline_config.json'
    for f in (gold_path, lexicon_path, config_path):
        if not f.is_file():
            raise FileNotFoundError(f'Not found: {f}')
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    components = cfg.get('components', {})
    fuzzy_on = components.get('fuzzy_matching', True)
    regex_on = components.get('regex_matching', True)
    morphology_on = components.get('morphology', True)
    category_guard_on = components.get('category_guard', True)
    sentiment_filter_on = components.get('stopword_sentiment_filter', True)
    taxonomy = cfg.get('taxonomy', 'evaluation16')

    sys.path.insert(0, str(repo / 'code'))
    from ate_pipeline_core import (  # ORIGINAL code, unmodified
        CORE_MAL_SUFFIXES, EVALUATION_CATEGORIES, BasicOnlyNormalizer,
        LexiconMatcher, Normalizer, category_specific_guard,
        clean_review_text, load_union, load_word_set, normalize_basic,
        parse_aspect_lexicon, surface_tokenize,
    )

    by_cat, entries = parse_aspect_lexicon(str(lexicon_path))
    if taxonomy == 'evaluation16':
        entries = [e for e in entries if e.category in EVALUATION_CATEGORIES]
    elif taxonomy != 'lexicon':
        raise ValueError(f'Unexpected taxonomy: {taxonomy}')

    matcher = LexiconMatcher(entries)
    resources = repo / 'resources'
    suffixes = load_union([str(resources / f) for f in
                           ('malayalam_suffix_1.txt', 'malayalam_suffixmorph.txt')])
    suffixes |= {normalize_basic(s) for s in CORE_MAL_SUFFIXES}
    base_normalizer = Normalizer(suffixes)
    if morphology_on and not base_normalizer.wordnet_available:
        raise RuntimeError('NLTK WordNet data unavailable. Run: py -m nltk.downloader wordnet omw-1.4')
    normalizer = base_normalizer if morphology_on else BasicOnlyNormalizer()

    stopwords = {normalize_basic(w) for w in load_word_set(str(resources / 'merged_stopwords.txt'))}
    sentiment_files = (
        'negative-words_english.txt', 'neg_word_malayalam.txt',
        'positive-words_english.txt', 'pos_words_malayalam.txt',
    )
    sentiment = {normalize_basic(w) for w in load_union([str(resources / f) for f in sentiment_files])}
    if not sentiment_filter_on:
        stopwords, sentiment = set(), set()

    df = pd.read_csv(gold_path)
    required = {'Review_Text', 'Word_Tokens', 'BIO_Tags'}
    if required - set(df.columns):
        raise ValueError(f'Gold file missing columns {sorted(required - set(df.columns))}')
    full_count = len(df)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        df = df.head(args.limit)
        print('SMOKE-TEST ONLY: DO NOT REPORT AS FULL CORPUS RESULTS', flush=True)
    else:
        if full_count != 1000:
            raise ValueError(f'Expected 1000 gold rows in original corpus, found {full_count}; review input first')

    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    match_type_counts = Counter()
    category_span_counts = Counter()
    tokenization_mismatches = []
    total_gold = 0
    reviews_with_gold = 0
    for i, record in df.iterrows():
        gold_tokens = [str(t) for t in as_list(record['Word_Tokens'], i, 'Word_Tokens')]
        tags = get_tags(record['BIO_Tags'], i)
        if len(gold_tokens) != len(tags):
            raise AssertionError(f'Row {i}: {len(gold_tokens)} tokens but {len(tags)} BIO tags')
        pipeline_tokens = surface_tokenize(clean_review_text(record['Review_Text']))
        if gold_tokens != pipeline_tokens:
            tokenization_mismatches.append(i)
            # Continuing would produce false-positive/false-negative span coverage.
            continue
        spans = strict_gold_spans(tags, i)
        total_gold += len(spans)
        reviews_with_gold += bool(spans)
        for _, _, category in spans:
            category_span_counts[category] += 1

        variants = [normalizer.variants(token) for token in pipeline_tokens]
        raw = matcher.match(pipeline_tokens, variants, stopwords, sentiment,
                            normalizer, float(cfg.get('fuzzy_cutoff', 0.75)))
        active = [c for c in raw if (fuzzy_on or c.match_type != 'fuzzy')
                  and (regex_on or c.match_type != 'regex')]
        match_type_counts.update(c.match_type for c in active)
        guarded = ([c for c in active if category_specific_guard(c, pipeline_tokens)]
                   if category_guard_on else active)
        keys_active = defaultdict(list)
        keys_guarded = defaultdict(list)
        keys_unfiltered = defaultdict(list)
        for c in active:
            keys_active[(c.start, c.end, c.category)].append(c)
        for c in guarded:
            keys_guarded[(c.start, c.end, c.category)].append(c)
            if not c.filtered:
                keys_unfiltered[(c.start, c.end, c.category)].append(c)

        for start, end, cat in spans:
            key = (start, end, cat)
            matches = keys_active.get(key, [])
            gmatches = keys_guarded.get(key, [])
            umatches = keys_unfiltered.get(key, [])
            rows.append({
                'original_row_index': int(i), 'gold_start': int(start), 'gold_end': int(end),
                'gold_category': cat, 'gold_surface': ' '.join(gold_tokens[start:end]),
                'candidate_before_context': int(bool(matches)),
                'candidate_after_category_guard': int(bool(gmatches)),
                'candidate_after_guard_unfiltered': int(bool(umatches)),
                'match_types_before_context': ';'.join(sorted({c.match_type for c in matches})),
                'contains_exact_candidate': int(any(c.match_type == 'exact' for c in matches)),
                'contains_fuzzy_candidate': int(any(c.match_type == 'fuzzy' for c in matches)),
                'contains_regex_candidate': int(any(c.match_type == 'regex' for c in matches)),
                'contains_heading_trigger': int(any(c.category_trigger for c in matches)),
            })
        if (i + 1) % 100 == 0:
            print(f'Processed {i + 1}/{len(df)} reviews; gold spans so far={total_gold}', flush=True)

    pd.DataFrame(rows).to_csv(outdir / 'gold_span_candidate_coverage_rows.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([r for r in rows if not r['candidate_before_context']]).to_csv(
        outdir / 'uncovered_gold_spans.csv', index=False, encoding='utf-8-sig')

    by_cat = []
    for cat in sorted(category_span_counts):
        subset = [r for r in rows if r['gold_category'] == cat]
        by_cat.append({'Category': cat, 'Gold_Spans': len(subset),
                       'Raw_Lexical_Covered': sum(r['candidate_before_context'] for r in subset),
                       'After_Category_Guard_Covered': sum(r['candidate_after_category_guard'] for r in subset)})
    pd.DataFrame(by_cat).to_csv(outdir / 'candidate_coverage_by_category.csv', index=False)
    result = {
        'scope': 'original evaluation_gold.csv, original matcher on identical pipeline/gold token sequences',
        'smoke_test_only': args.limit is not None,
        'evaluation_rows_examined': len(df),
        'gold_rows_with_at_least_one_span': int(reviews_with_gold),
        'gold_spans_in_aligned_rows': int(total_gold),
        'rows_with_tokenization_mismatch': len(tokenization_mismatches),
        'mismatched_row_indices_first_20': tokenization_mismatches[:20],
        'configured_components': {'fuzzy_matching': fuzzy_on, 'regex_matching': regex_on,
                                  'morphology': morphology_on, 'category_guard': category_guard_on,
                                  'stopword_sentiment_filter': sentiment_filter_on, 'taxonomy': taxonomy},
        'raw_lexical_candidate_pre_context': summarize_span_count(rows, 'candidate_before_context', total_gold),
        'post_category_guard_pre_context': summarize_span_count(rows, 'candidate_after_category_guard', total_gold),
        'post_guard_unfiltered_only_pre_context': summarize_span_count(rows, 'candidate_after_guard_unfiltered', total_gold),
        'active_candidate_match_types': dict(match_type_counts),
        'gold_category_span_counts': dict(category_span_counts),
        'sha256': {'gold': hash_file(gold_path), 'lexicon': hash_file(lexicon_path), 'config': hash_file(config_path)},
        'interpretation': [
            'Strict category and token-boundary match is required for a gold span to be considered covered.',
            'This is candidate-stage coverage before frozen-mBERT contextual filtering and final overlap selection.',
            'Active regex configuration is respected; regex patterns are never removed from the original resource.',
            'Candidate coverage is an upper bound on recall achievable from these lexical candidates, not final pipeline recall.',
            'A historical Table 5 value is not validated merely because this coverage is close to it.',
            'Historical lexicon provenance and pruned-lexicon results are NOT established by this script.',
        ],
    }
    (outdir / 'gold_candidate_coverage_summary.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    print('Saved:', outdir / 'gold_span_candidate_coverage_rows.csv')
    print('Saved:', outdir / 'uncovered_gold_spans.csv')
    print('Saved:', outdir / 'candidate_coverage_by_category.csv')
    print('Saved:', outdir / 'gold_candidate_coverage_summary.json')
    print('No model inference, lexicon editing, or configuration changes made.')


if __name__ == '__main__':
    main()
