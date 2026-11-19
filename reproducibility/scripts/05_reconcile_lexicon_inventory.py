"""Read-only lexicon counting audit for Reviewer 2, Comment 1.

Usage (PowerShell, from IDT_R3\\R3):
 py .\\05_reconcile_lexicon_inventory.py --repo "..\\GitHub_R2" --exposure "R2_checks\\exposure\\current_matcher_exposure_summary.json" --outdir "R2_checks\\lexicon_inventory"

This script does NOT change the pipeline, lexical file, thresholds, or evaluation.
It distinguishes counts for raw lexical lines from operational parser entries.
Exposure in 04_diagnostic_lexicon_exposure.py is matcher- and fuzzy-based; it is
NOT an exact-form-occurrence audit and cannot establish historical independence.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from collections import Counter

HEADERS = {
    "memory", "screen", "display", "ui-ux", "camera", "phone", "os",
    "tablet", "laptop", "battery", "software", "processor", "applications",
    "price", "speaker", "network", "hardware", "design", "general", "game",
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--exposure", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("R2_checks/lexicon_inventory"))
    a = ap.parse_args()
    root = a.repo.resolve()
    lex = root / "resources" / "aspect_lexicon.txt"
    assert lex.is_file(), f"Lexicon not found: {lex}"
    out = a.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    raw_lines = [line.strip() for line in lex.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    header_lines = [s for s in raw_lines if s.endswith("-") and s[:-1].casefold() in HEADERS]
    lexical_lines = [s for s in raw_lines if not (s.endswith("-") and s[:-1].casefold() in HEADERS)]

    # Import the repository's ORIGINAL lexicon parser, without any replacement.
    sys.path.insert(0, str(root / "code"))
    from ate_pipeline_core import parse_aspect_lexicon
    _, parsed = parse_aspect_lexicon(str(lex))
    literal = [e for e in parsed if not e.regex and not e.category_trigger]
    triggers = [e for e in parsed if e.category_trigger]
    regexes = [e for e in parsed if e.regex]

    result = {
        "lexicon_sha256": file_hash(lex),
        "raw_nonblank_lines_including_headers": len(raw_lines),
        "raw_category_heading_lines": len(header_lines),
        "raw_lexical_entry_lines_excluding_headers": len(lexical_lines),
        "raw_distinct_casefolded_entry_forms_including_regex": len({s.casefold() for s in lexical_lines}),
        "operational_parser_entries_after_category_dedup": len(parsed),
        "operational_parser_literal_entries_across_categories": len(literal),
        "operational_parser_distinct_normalized_literal_forms": len({e.normalized for e in literal}),
        "operational_parser_category_triggers": len(triggers),
        "operational_parser_regex_entries": len(regexes),
        "lexicon_categories": sorted({e.original_category for e in parsed}),
        "per_output_category_parsed_entries": dict(sorted(Counter(e.category for e in parsed).items())),
        "notes": [
            "Raw casefolded lexical entries may include regex expressions.",
            "Operational literal forms exclude regex entries and heading-trigger entries; category aliasing/dedup and lexical normalization apply.",
            "The 04 diagnostic counts candidate match exposure, including fuzzy matching (cutoff 0.75), not literal exact occurrences.",
            "None of these inventory counts proves whether the historical lexicon was selected independently of the evaluation corpus.",
        ],
    }
    if a.exposure:
        path = a.exposure.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Exposure diagnostic not found: {path}")
        exposure = json.loads(path.read_text(encoding="utf-8"))
        if exposure.get("lexicon_sha256") != result["lexicon_sha256"]:
            raise ValueError("Lexicon SHA-256 mismatch: exposure diagnostic used another lexicon!")
        result["exposure_diagnostic"] = {
            "method": exposure.get("method"),
            "source_rows": exposure.get("full_source_rows"),
            "evaluation_rows": exposure.get("evaluation_rows"),
            "source_text_disjoint_rows": exposure.get("text_disjoint_source_rows"),
            "distribution": exposure.get("distribution"),
            "distribution_sum": sum(exposure.get("distribution", {}).values()),
        }
        expected = result["operational_parser_distinct_normalized_literal_forms"]
        if result["exposure_diagnostic"]["distribution_sum"] != expected:
            raise ValueError(f"Exposure classes do not sum to parser forms: {expected}")

    target = out / "lexicon_inventory_definition_audit.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {target}")
    print("No extraction logic or lexicon contents were changed.")

if __name__ == "__main__":
    main()
