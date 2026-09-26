#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-component-at-a-time ablation for the configured weak ATE pipeline.

Only components active in the supplied final configuration are ablated. The
contribution of a component is defined as Full_F1 - Ablated_F1, so a positive
value directly indicates that the component improves the configured system.
Negative or neutral contributions are retained rather than filtered or hidden.
No fixed evaluation size or expected metric is encoded in this script.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from evaluate_ate import metric_tables, sequences

COMPONENTS = [
    ("morphology", "components", "morphology"),
    ("stopword/sentiment filtering", "components", "stopword_sentiment_filter"),
    ("fuzzy lexical matching", "components", "fuzzy_matching"),
    ("regex lexical matching", "components", "regex_matching"),
    ("contextual filtering", "components", "contextual_filtering"),
    ("category guard", "components", "category_guard"),
    ("evidence-aware overlap resolution", "components", "evidence_overlap_resolution"),
    ("semantic discovery", "semantic_discovery", "enabled"),
    ("neighbor extension", "neighbor_extension", "enabled"),
]


def parse_obj(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    return ast.literal_eval(str(x))


def enabled(cfg, section, key):
    if section == "components":
        return bool(cfg.get(section, {}).get(key, True))
    return bool(cfg.get(section, {}).get(key, False))


def disable(cfg, section, key):
    out = copy.deepcopy(cfg)
    out.setdefault(section, {})[key] = False
    return out


def run_generator(args, config_path, input_path, output_path):
    cmd = [
        sys.executable, args.generator,
        "--config", str(config_path),
        "--input", str(input_path),
        "--output", str(output_path),
        "--lexicon", args.lexicon,
        "--stopwords", args.stopwords,
    ]
    for p in args.suffix_file:
        cmd.extend(["--suffix-file", p])
    for p in args.sentiment_file:
        cmd.extend(["--sentiment-file", p])
    if args.device:
        cmd.extend(["--device", args.device])
    subprocess.run(cmd, check=True)


def micro_metrics(gold, pred, manifest):
    y_true, y_pred = sequences(gold, pred, manifest)
    summary, _, _ = metric_tables(y_true, y_pred)
    row = summary[summary["Metric"] == "micro avg"].iloc[0]
    return {
        "Precision": float(row["Precision"]),
        "Recall": float(row["Recall"]),
        "F1": float(row["F1"]),
        "Support": int(row["Support"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator", required=True, help="Path to generate_ate_bio.py")
    ap.add_argument("--config", required=True, help="Final frozen pipeline JSON")
    ap.add_argument("--input", required=True, help="Review input corresponding row-for-row to gold")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--manifest", required=True, help="Common evaluation manifest")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--suffix-file", action="append", required=True)
    ap.add_argument("--stopwords", required=True)
    ap.add_argument("--sentiment-file", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = pd.read_csv(args.input)
    manifest = pd.read_csv(args.manifest)
    if "Original_Row_Index" not in manifest.columns:
        raise KeyError("Evaluation manifest requires Original_Row_Index")

    idx = manifest["Original_Row_Index"].astype(int).tolist()
    if min(idx) < 0 or max(idx) >= len(source):
        raise IndexError("Evaluation manifest contains an input row index outside the source CSV")
    subset = source.iloc[idx].reset_index(drop=True)
    subset_path = out / "evaluation_input.csv"
    subset.to_csv(subset_path, index=False)

    variants = [("Full pipeline", cfg)]
    for display, section, key in COMPONENTS:
        if enabled(cfg, section, key):
            variants.append((f"Without {display}", disable(cfg, section, key)))

    rows = []
    for number, (name, variant_cfg) in enumerate(variants):
        cfg_path = out / f"config_{number:02d}.json"
        pred_path = out / f"prediction_{number:02d}.csv"
        cfg_path.write_text(json.dumps(variant_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        run_generator(args, cfg_path, subset_path, pred_path)
        m = micro_metrics(args.gold, pred_path, args.manifest)
        rows.append({"Configuration": name, **m, "Prediction_File": str(pred_path)})

    results = pd.DataFrame(rows)
    full_f1 = float(results.loc[results["Configuration"] == "Full pipeline", "F1"].iloc[0])
    results["Contribution_F1"] = full_f1 - results["F1"]
    results.loc[results["Configuration"] == "Full pipeline", "Contribution_F1"] = 0.0
    results["Contribution"] = results["Contribution_F1"].apply(
        lambda x: "positive" if x > 0 else ("neutral" if x == 0 else "negative")
    )
    results.to_csv(out / "component_ablation.csv", index=False)
    print(results[["Configuration", "Precision", "Recall", "F1", "Contribution_F1", "Contribution"]].to_string(index=False))


if __name__ == "__main__":
    main()
