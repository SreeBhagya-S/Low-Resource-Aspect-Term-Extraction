#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Read-only Reviewer 2 / Table 5 reconciliation of script 04 vs script 06.

Usage (PowerShell, from C:\\Users\\sreeb\\Desktop\\IDT_R3\\R3):
  py 07_reconcile_exposure_discrepancy.py `
    --script04 "R2_checks\exposure\current_matcher_literal_exposure.csv" `
    --script06 "R2_checks\exact_fuzzy\exact_vs_fuzzy_form_exposure.csv" `
    --repo "..\GitHub_R2" `
    --outdir "R2_checks\exposure_discrepancy"

NO model inference, retraining, matching, editing of resources, or changes to metrics.
Only compares the two CSV inventories already generated and reads the released
pipeline configuration. It is NOT proof of lexicon independence and does NOT
calculate gold-span candidate recall ceilings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

CLASSES = ("both", "source_only", "evaluation_only", "neither")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for part in iter(lambda: file.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def require(df, fields, path):
    missing = set(fields).difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if df["normalized_literal"].isna().any():
        raise ValueError(f"Empty normalized literal in {path}")
    if df["normalized_literal"].duplicated().any():
        raise ValueError(f"Repeated normalized literal in {path}")


def freq(series):
    c = Counter(series)
    return {k: int(c[k]) for k in CLASSES}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--script04", type=Path, required=True)
    p.add_argument("--script06", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("R2_checks/exposure_discrepancy"))
    args = p.parse_args()
    path04, path06, repo, outdir = [x.resolve() for x in (args.script04, args.script06, args.repo, args.outdir)]
    for f in (path04, path06):
        if not f.is_file():
            raise FileNotFoundError(f"Cannot find {f}")
    configfile = repo / "config" / "pipeline_config.json"
    lexfile = repo / "resources" / "aspect_lexicon.txt"
    if not configfile.is_file() or not lexfile.is_file():
        raise FileNotFoundError("Cannot find config/pipeline_config.json or resources/aspect_lexicon.txt inside --repo")
    outdir.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(path04, encoding="utf-8-sig", dtype=str).fillna("")
    b = pd.read_csv(path06, encoding="utf-8-sig", dtype=str).fillna("")
    require(a, ("normalized_literal", "exposure_class"), path04)
    require(b, ("normalized_literal", "script04_compatible_class", "operational_exact_class", "exact_plus_fuzzy_class"), path06)
    for name, df, col in (("script04", a, "exposure_class"), ("script06", b, "script04_compatible_class")):
        unexpected = sorted(set(df[col]).difference(CLASSES))
        if unexpected:
            raise ValueError(f"Unexpected status in {name}: {unexpected}")

    idx_a = a.set_index("normalized_literal")
    idx_b = b.set_index("normalized_literal")
    only_a = sorted(set(idx_a.index) - set(idx_b.index))
    only_b = sorted(set(idx_b.index) - set(idx_a.index))
    common = sorted(set(idx_a.index) & set(idx_b.index))

    details = []
    for term in common:
        row4 = idx_a.loc[term]
        row6 = idx_b.loc[term]
        if row4["exposure_class"] != row6["script04_compatible_class"]:
            rec = {
                "normalized_literal": term,
                "script04_class": row4["exposure_class"],
                "script06_script04_compatible_class": row6["script04_compatible_class"],
                "script06_exact_class": row6["operational_exact_class"],
                "script06_exact_plus_fuzzy_class": row6["exact_plus_fuzzy_class"],
            }
            for col in ("exposed_in_evaluation", "exposed_in_text_disjoint_source"):
                if col in idx_a.columns:
                    rec["script04_" + col] = row4[col]
            for col in ("exact_eval", "exact_source", "fuzzy_eval", "fuzzy_source"):
                if col in idx_b.columns:
                    rec["script06_" + col] = row6[col]
            details.append(rec)

    cfg = json.loads(configfile.read_text(encoding="utf-8"))
    comp = cfg.get("components", {})
    summary = {
        "purpose": "Check mismatching per-form classifications, not infer historical provenance",
        "source_file_hashes": {
            "script04_csv_sha256": sha256(path04),
            "script06_csv_sha256": sha256(path06),
            "original_lexicon_sha256": sha256(lexfile),
            "pipeline_config_sha256": sha256(configfile),
        },
        "forms_script04": int(len(a)),
        "forms_script06": int(len(b)),
        "form_names_only_in_script04": only_a,
        "form_names_only_in_script06": only_b,
        "script04_all_match_types_distribution": freq(a["exposure_class"]),
        "script06_compatible_distribution": freq(b["script04_compatible_class"]),
        "script06_operational_exact_distribution": freq(b["operational_exact_class"]),
        "script06_exact_plus_fuzzy_distribution": freq(b["exact_plus_fuzzy_class"]),
        "same_form_different_class_count": len(details),
        "different_form_details": details,
        "active_pipeline_config": {
            "regex_matching": comp.get("regex_matching", None),
            "fuzzy_matching": comp.get("fuzzy_matching", None),
            "fuzzy_cutoff": cfg.get("fuzzy_cutoff", None),
            "taxonomy": cfg.get("taxonomy", None),
        },
        "interpretation": [
            "The script06 candidate_match_types raw counts include regex even if the active pipeline config disables regex matching.",
            "The 'operational_exact' column uses normalized matcher candidates, not literal substring occurrences.",
            "Disagreement between 04 and 06 on the same form needs inspection, potentially including nondeterministic selection among normalization variants in original fuzzy matching.",
            "All exposures are descriptive. They cannot establish that lexicon selection was independent of evaluation reviews.",
            "The legacy Table 5 recall ceilings and pruned-model metrics are not computed by this script.",
        ],
    }

    pd.DataFrame(details, columns=[
        "normalized_literal", "script04_class", "script06_script04_compatible_class",
        "script06_exact_class", "script06_exact_plus_fuzzy_class",
        "script04_exposed_in_evaluation", "script04_exposed_in_text_disjoint_source",
        "script06_exact_eval", "script06_exact_source", "script06_fuzzy_eval", "script06_fuzzy_source"
    ]).to_csv(outdir / "discrepant_forms.csv", index=False, encoding="utf-8-sig")
    (outdir / "exposure_reconciliation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {outdir / 'discrepant_forms.csv'}")
    print(f"Saved: {outdir / 'exposure_reconciliation.json'}")
    print("No extraction logic, thresholds, predictions, or lexicon contents changed.")


if __name__ == "__main__":
    main()
