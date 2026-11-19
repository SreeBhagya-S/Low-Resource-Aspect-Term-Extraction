#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer 2: create an AUDITABLE, POST-HOC lexicon sensitivity resource.

This script makes a separate copy of the ORIGINAL lexical resource with the
17 NORMALIZED FORMS classified as evaluation-only by Script 06's OPERATIONAL
EXACT matching removed. It never edits the original lexicon, source code,
model configuration, gold labels, or predictions. It does NOT run inference.

Windows PowerShell (from C:\\Users\\sreeb\\Desktop\\IDT_R3\\R3):
  py 10_prepare_pruned_lexicon.py `
    --repo "..\\GitHub_R2" `
    --exposure "R2_checks\\exact_fuzzy\\exact_vs_fuzzy_form_exposure.csv" `
    --outdir "R2_checks\\pruning"

NB: This is post-hoc sensitivity analysis, NOT a demonstration that the
original lexicon was built independently of the evaluation corpus.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--exposure', type=Path, required=True,
                        help='Script 06 exact_vs_fuzzy_form_exposure.csv')
    parser.add_argument('--outdir', type=Path, default=Path('R2_checks/pruning'))
    args = parser.parse_args()
    repo, exposure, outdir = args.repo.resolve(), args.exposure.resolve(), args.outdir.resolve()
    lexicon = repo / 'resources' / 'aspect_lexicon.txt'
    config = repo / 'config' / 'pipeline_config.json'
    for path in (lexicon, config, exposure):
        if not path.is_file():
            raise FileNotFoundError(f'Required file not found: {path}')
    if lexicon.is_relative_to(outdir):
        raise ValueError('Output directory cannot contain/overwrite the original lexicon')

    # Import the original parser: neither it nor the matcher is changed.
    sys.path.insert(0, str(repo / 'code'))
    from ate_pipeline_core import (  # type: ignore
        KNOWN_CATEGORY_HEADERS, looks_like_regex, normalize_basic,
        parse_aspect_lexicon, unicode_nfc,
    )

    with exposure.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        need = {'normalized_literal', 'operational_exact_class'}
        if not need.issubset(reader.fieldnames or ()):
            raise ValueError('Exposure CSV must be the output of Script 06, with columns: '
                             + ', '.join(sorted(need)))
        records = list(reader)
    if len(records) != 517 or len({r['normalized_literal'] for r in records}) != 517:
        raise ValueError(f'Expected 517 distinct operational literal forms, got {len(records)} rows')
    removed = {r['normalized_literal'] for r in records
               if r['operational_exact_class'] == 'evaluation_only'}
    if len(removed) != 17:
        raise ValueError(f'Expected exactly 17 evaluation-only forms; found {len(removed)}')

    _, old_entries = parse_aspect_lexicon(str(lexicon))
    old_forms = {e.normalized for e in old_entries if not e.regex and not e.category_trigger}
    if len(old_forms) != 517 or old_forms != {r['normalized_literal'] for r in records}:
        raise ValueError('Parser inventory does not match Script 06 CSV. STOP: check repo and exposure versions.')

    # Preserve category headings, regex, other entries, original order/encoding.
    raw_lines = lexicon.read_text(encoding='utf-8-sig').splitlines(keepends=True)
    kept = []
    removed_lines = []
    current_category = None
    for line_no, original_line in enumerate(raw_lines, 1):
        raw = unicode_nfc(original_line).strip()
        if not raw:
            kept.append(original_line)
            continue
        candidate_header = normalize_basic(raw[:-1]) if raw.endswith('-') else ''
        if raw.endswith('-') and candidate_header in KNOWN_CATEGORY_HEADERS:
            current_category = candidate_header
            kept.append(original_line)
            continue
        if (current_category is not None and not looks_like_regex(raw)
                and normalize_basic(raw) in removed):
            removed_lines.append({'line': line_no, 'category_heading': current_category,
                                  'raw_entry': raw, 'normalized_literal': normalize_basic(raw)})
        else:
            kept.append(original_line)
    if {r['normalized_literal'] for r in removed_lines} != removed:
        unaccounted = sorted(removed - {r['normalized_literal'] for r in removed_lines})
        raise ValueError(f'Cannot safely map all 17 forms to original lexicon lines: {unaccounted}')

    outdir.mkdir(parents=True, exist_ok=True)
    pruned_path = outdir / 'aspect_lexicon_pruned17.txt'
    if pruned_path.resolve() == lexicon.resolve():
        raise ValueError('Refusing to overwrite original lexicon')
    pruned_path.write_text(''.join(kept), encoding='utf-8')

    _, new_entries = parse_aspect_lexicon(str(pruned_path))
    new_forms = {e.normalized for e in new_entries if not e.regex and not e.category_trigger}
    if (old_forms - new_forms) != removed or (new_forms - old_forms) or len(new_forms) != 500:
        pruned_path.unlink(missing_ok=True)
        raise AssertionError('Pruning failed parser-inventory verification. Pruned file deleted.')
    before_other = sorted((e.category, e.normalized, e.regex, e.category_trigger)
                          for e in old_entries if e.regex or e.category_trigger)
    after_other = sorted((e.category, e.normalized, e.regex, e.category_trigger)
                         for e in new_entries if e.regex or e.category_trigger)
    if before_other != after_other:
        pruned_path.unlink(missing_ok=True)
        raise AssertionError('Regex or category-heading triggers unexpectedly changed. Pruned file deleted.')

    removed_csv = outdir / 'pruned17_removed_lines.csv'
    with removed_csv.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['line', 'category_heading', 'raw_entry', 'normalized_literal'])
        writer.writeheader()
        writer.writerows(removed_lines)

    manifest = {
        'description': 'Post-hoc sensitivity resource; NOT leakage-free resource construction',
        'selection_definition': 'Script 06 operational_exact_class=evaluation_only',
        'original_lexicon': str(lexicon), 'original_sha256': sha256(lexicon),
        'exposure_csv': str(exposure), 'exposure_sha256': sha256(exposure),
        'original_config': str(config), 'config_sha256': sha256(config),
        'pruned_lexicon': str(pruned_path), 'pruned_sha256': sha256(pruned_path),
        'normalized_literal_forms_before': len(old_forms),
        'normalized_literal_forms_removed': len(removed),
        'normalized_literal_forms_after': len(new_forms),
        'raw_lexical_lines_removed': len(removed_lines),
        'removed_forms': sorted(removed),
        'regex_and_heading_triggers_preserved': True,
        'note': ('Selection based on evaluation-set occurrence is post-hoc. '
                 'Results on the final test are only a sensitivity diagnostic, '
                 'not proof of a clean train/test separation.'),
    }
    (outdir / 'pruning_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('removed_forms','note')},
                     ensure_ascii=False, indent=2))
    print(f'Pruned copy: {pruned_path}')
    print(f'Removed line audit: {removed_csv}')
    print('Original lexicon, pipeline source, and config unchanged. No inference performed.')


if __name__ == '__main__':
    main()
