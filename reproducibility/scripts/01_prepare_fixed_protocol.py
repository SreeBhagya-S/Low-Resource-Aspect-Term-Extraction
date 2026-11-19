"""Prepare auditable inputs for the repository's ORIGINAL ATE implementation.

Does NOT modify the pipeline, gold annotations, lexicon, thresholds or review assignments.
Creates 200-row inference input in the exact fixed-test manifest order, and Role-based
manifests required by the existing baseline scripts.

Run from any folder:
  py 01_prepare_fixed_protocol.py --repo "Low-Resource-Aspect-Term-Extraction-main" --outdir "R2_checks"
"""
import argparse
import ast
import hashlib
import json
import unicodedata
from pathlib import Path
import pandas as pd


def normalize(x):
    return ' '.join(unicodedata.normalize('NFKC', str(x)).casefold().split())


def parse_list(x):
    obj = ast.literal_eval(str(x))
    if not isinstance(obj, (list, tuple)):
        raise ValueError('Expected a serialized list')
    return list(obj)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--outdir', default='R2_checks', type=Path)
    args = ap.parse_args()
    p = args.repo.resolve()
    o = args.outdir.resolve()
    o.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(p/'data/evaluation_input.csv')
    gold = pd.read_csv(p/'data/evaluation_gold.csv')
    manifest = pd.read_csv(p/'data/fixed_test_manifest_seed42.csv')
    protocol = pd.read_csv(p/'data/dual_regime_protocol_manifest.csv')
    assert len(source) == len(gold) == len(protocol) == 1000, 'Expected 1,000 source/gold/protocol rows'
    assert len(manifest) == 200, 'Expected fixed 200-row test manifest'
    idx = manifest.Original_Row_Index.astype(int).tolist()
    assert len(set(idx)) == len(idx), 'Duplicate final test indices'
    assert min(idx) >= 0 and max(idx) < len(source), 'Test index out of range'
    test_indices = set(idx)
    from_protocol = set(protocol.loc[protocol.Outer_Role == 'final_test','Original_Row_Index'].astype(int))
    assert test_indices == from_protocol, 'Fixed manifest is not the protocol final test'
    assert source.Review_Text.map(normalize).equals(gold.Review_Text.map(normalize)), 'Source vs gold mismatch'

    for j, i in enumerate(idx):
        m = manifest.iloc[j]
        assert normalize(source.iloc[i].Review_Text) == normalize(m.Review_Text), f'Raw text mismatch: manifest row {j}'
        expected_tokens = [str(t) for t in parse_list(m.Word_Tokens_Evaluated)]
        gold_tokens = [str(t) for t in parse_list(gold.iloc[i].Word_Tokens)]
        assert gold_tokens[:len(expected_tokens)] == expected_tokens, f'Gold token mismatch: manifest row {j}'
        gold_tags = parse_list(m.Gold_BIO)
        assert len(gold_tags) == len(expected_tokens), f'Gold tag count mismatch: manifest row {j}'

    test = source.iloc[idx].reset_index(drop=True)
    test.to_csv(o/'test_input_200.csv', index=False, encoding='utf-8-sig')
    # Generate baseline compatibility files from the existing protocol; identical memberships.
    for regime, column in [('low_resource','Low_Resource_Role'),('full_supervision','Full_Supervised_Role')]:
        mapping = {'train':'supervised_train', 'validation':'supervised_validation'}
        chosen = protocol.loc[protocol[column].isin(mapping)].copy()
        result = chosen[['Original_Row_Index']].copy()
        result['Role'] = chosen[column].map(mapping).values
        result.to_csv(o/f'{regime}_baseline_manifest.csv', index=False)
        expected = (64,16) if regime == 'low_resource' else (720,80)
        actual = (int((result.Role=='supervised_train').sum()),int((result.Role=='supervised_validation').sum()))
        assert actual == expected, f'{regime} count mismatch: {actual}'
        assert not (set(result.Original_Row_Index.astype(int)) & test_indices), 'Development/test contamination'

    audit = {
        'source_reviews':len(source), 'test_reviews':len(test),
        'test_indices_first_ten':idx[:10],
        'low_resource':{'train':64,'validation':16},
        'full_supervision':{'train':720,'validation':80},
        'test_order_manifest':str((p/'data/fixed_test_manifest_seed42.csv').resolve()),
        'files_sha256':{str(f.relative_to(p)):digest(f) for f in [p/'data/evaluation_input.csv',p/'data/evaluation_gold.csv',p/'data/fixed_test_manifest_seed42.csv',p/'data/dual_regime_protocol_manifest.csv',p/'config/pipeline_config.json',p/'resources/aspect_lexicon.txt']},
    }
    (o/'protocol_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: 1,000 aligned annotated reviews; fixed 200 test; 80 low-resource / 800 full-supervision development.')
    print('Saved:',o/'test_input_200.csv')
    print('Saved:',o/'low_resource_baseline_manifest.csv')
    print('Saved:',o/'full_supervision_baseline_manifest.csv')
    print('No pipeline logic, lexicon, thresholds, or assignments were changed.')


if __name__ == '__main__':
    main()
