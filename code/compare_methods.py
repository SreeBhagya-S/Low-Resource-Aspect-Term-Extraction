#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare ATE systems on one common gold evaluation manifest.

Every metric is recomputed from prediction files. The script contains no stored
scores, winners, support values, or system-specific target thresholds.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evaluate_ate import metric_tables, sequences


def parse_system(spec: str):
    name, sep, path = spec.partition("=")
    if not sep or not name.strip() or not path.strip():
        raise ValueError(f"Invalid --system specification {spec!r}; use NAME=PATH")
    return name.strip(), path.strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--system", action="append", required=True,
                    help="Repeat as NAME=prediction.csv")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    per_class_rows = []

    for spec in args.system:
        name, path = parse_system(spec)
        y_true, y_pred = sequences(args.gold, path, args.manifest)
        summary, per_class, _ = metric_tables(y_true, y_pred)
        summary.insert(0, "Method", name)
        per_class.insert(0, "Method", name)
        summary_rows.append(summary)
        per_class_rows.append(per_class)

    summary = pd.concat(summary_rows, ignore_index=True)
    per_class = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    summary.to_csv(out / "method_comparison_all_averages.csv", index=False)
    if not per_class.empty:
        per_class.to_csv(out / "method_comparison_per_category.csv", index=False)

    micro = summary[summary["Metric"] == "micro avg"].copy()
    micro = micro.sort_values("F1", ascending=False).reset_index(drop=True)
    micro.to_csv(out / "method_comparison_micro.csv", index=False)
    print(micro.to_string(index=False))


if __name__ == "__main__":
    main()
