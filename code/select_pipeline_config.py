#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select a pipeline configuration using development data only.

The search space is supplied as JSON rather than embedded in source. Each
configuration is evaluated with strict span-level IOB2 micro-F1 on the selected
development manifest. The final configuration and the complete search table are
saved for auditability.
"""
from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from evaluate_ate import metric_tables, sequences


def parse_obj(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    return ast.literal_eval(str(x))


def apply_filters(df: pd.DataFrame, filters):
    out = df.copy()
    for item in filters:
        key, sep, raw = item.partition("=")
        if not sep or key not in out.columns:
            raise ValueError(f"Invalid manifest filter: {item}")
        numeric = pd.to_numeric(out[key], errors="coerce")
        try:
            value = float(raw)
            mask = numeric == value
            if mask.any():
                out = out[mask]; continue
        except ValueError:
            pass
        out = out[out[key].astype(str) == raw]
    return out.copy()


def set_dotted(cfg, dotted, value):
    parts = dotted.split(".")
    cur = cfg
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def build_configs(search_spec):
    base = search_spec["base_config"]
    grid = search_spec.get("grid", {})
    if not grid:
        return [copy.deepcopy(base)]
    keys = list(grid)
    configs = []
    for values in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        for k, v in zip(keys, values):
            set_dotted(cfg, k, v)
        configs.append(cfg)
    return configs


def run_generator(args, cfg_path, input_path, output_path):
    cmd = [sys.executable, args.generator,
           "--config", str(cfg_path), "--input", str(input_path), "--output", str(output_path),
           "--lexicon", args.lexicon, "--stopwords", args.stopwords]
    for p in args.suffix_file: cmd.extend(["--suffix-file", p])
    for p in args.sentiment_file: cmd.extend(["--sentiment-file", p])
    if args.device: cmd.extend(["--device", args.device])
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--search-space", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--manifest-filter", action="append", default=[])
    ap.add_argument("--validation-role", default="supervised_validation")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--suffix-file", action="append", required=True)
    ap.add_argument("--stopwords", required=True)
    ap.add_argument("--sentiment-file", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.input); gold = pd.read_csv(args.gold)
    split = apply_filters(pd.read_csv(args.split_manifest), args.manifest_filter)
    idx = split.loc[split["Role"].astype(str) == args.validation_role, "Original_Row_Index"].astype(int).tolist()
    if not idx: raise ValueError("No validation rows selected from split manifest")

    subset = source.iloc[idx].reset_index(drop=True)
    subset_path = out / "validation_input.csv"; subset.to_csv(subset_path, index=False)

    # Build a generic validation manifest from the gold file. No support count is assumed.
    if not {"Word_Tokens", "BIO_Tags"}.issubset(gold.columns):
        raise KeyError("Gold CSV requires Word_Tokens and BIO_Tags")
    val_manifest = pd.DataFrame({
        "Original_Row_Index": idx,
        "Word_Tokens_Evaluated": [gold.iloc[i]["Word_Tokens"] for i in idx],
        "Gold_BIO": [[str(tag) for _, tag in parse_obj(gold.iloc[i]["BIO_Tags"])] for i in idx],
    })
    val_manifest_path = out / "validation_manifest.csv"; val_manifest.to_csv(val_manifest_path, index=False)

    spec = json.loads(Path(args.search_space).read_text(encoding="utf-8"))
    configs = build_configs(spec)
    rows = []
    for i, cfg in enumerate(configs):
        cfg_path = out / f"candidate_{i:04d}.json"
        pred_path = out / f"candidate_{i:04d}.csv"
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        run_generator(args, cfg_path, subset_path, pred_path)
        yt, yp = sequences(args.gold, str(pred_path), str(val_manifest_path))
        summary, _, _ = metric_tables(yt, yp)
        micro = summary[summary["Metric"] == "micro avg"].iloc[0]
        rows.append({"Config_ID": i, "Precision": float(micro["Precision"]),
                     "Recall": float(micro["Recall"]), "F1": float(micro["F1"]),
                     "Support": int(micro["Support"]), "Config_File": str(cfg_path)})

    results = pd.DataFrame(rows).sort_values(
        ["F1", "Precision", "Recall", "Config_ID"], ascending=[False, False, False, True]
    ).reset_index(drop=True)
    results.to_csv(out / "configuration_search.csv", index=False)
    best = json.loads(Path(results.iloc[0]["Config_File"]).read_text(encoding="utf-8"))
    (out / "selected_pipeline_config.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    print(results.head(min(len(results), 20)).to_string(index=False))
    print("Selected configuration saved to:", out / "selected_pipeline_config.json")


if __name__ == "__main__":
    main()
