#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict span-level IOB2 evaluation for ATE predictions.

The evaluator contains no dataset-size assumptions and no expected metric values.
It can score the full gold corpus or a manifest-defined subset/prefix shared with
other systems.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import unicodedata
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
SPACE_RE = re.compile(r"\s+")


def parse_obj(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    return ast.literal_eval(str(x))


def norm_text(x) -> str:
    s = unicodedata.normalize("NFC", str(x))
    s = ZERO_WIDTH_RE.sub("", s)
    return SPACE_RE.sub(" ", s).strip()


def tags_from_cell(value) -> List[str]:
    obj = parse_obj(value)
    if not obj:
        return []
    first = obj[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return [str(x[1]) for x in obj]
    return [str(x) for x in obj]


def tokens_from_cell(value) -> List[str]:
    return [str(x) for x in parse_obj(value)]


def load_prediction(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "Word_Tokens" in df.columns:
        df["_tokens"] = df["Word_Tokens"].apply(tokens_from_cell)
    elif "Word_Tokens_Evaluated" in df.columns:
        df["_tokens"] = df["Word_Tokens_Evaluated"].apply(tokens_from_cell)
    else:
        raise KeyError("Prediction CSV requires Word_Tokens or Word_Tokens_Evaluated")

    if "BIO_Tags" in df.columns:
        df["_tags"] = df["BIO_Tags"].apply(tags_from_cell)
    elif "Predicted_BIO" in df.columns:
        df["_tags"] = df["Predicted_BIO"].apply(tags_from_cell)
    else:
        raise KeyError("Prediction CSV requires BIO_Tags or Predicted_BIO")
    return df


def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if "Word_Tokens" not in df.columns or "BIO_Tags" not in df.columns:
        raise KeyError("Gold CSV requires Word_Tokens and BIO_Tags")
    df["_tokens"] = df["Word_Tokens"].apply(tokens_from_cell)
    df["_tags"] = df["BIO_Tags"].apply(tags_from_cell)
    for i, row in df.iterrows():
        if len(row["_tokens"]) != len(row["_tags"]):
            raise AssertionError(f"Gold token/tag length mismatch at row {i}")
    return df


def sequences(gold_path: str, pred_path: str, manifest_path: str | None):
    gold = load_gold(gold_path)
    pred = load_prediction(pred_path)
    y_true, y_pred = [], []

    if manifest_path:
        manifest = pd.read_csv(manifest_path)
        required = {"Original_Row_Index", "Word_Tokens_Evaluated", "Gold_BIO"}
        missing = required - set(manifest.columns)
        if missing:
            raise KeyError(f"Manifest missing columns: {sorted(missing)}")
        if len(pred) != len(manifest):
            raise ValueError("Prediction and evaluation-manifest row counts differ")

        for j, m in manifest.reset_index(drop=True).iterrows():
            idx = int(m["Original_Row_Index"])
            eval_tokens = tokens_from_cell(m["Word_Tokens_Evaluated"])
            gold_tags = tags_from_cell(m["Gold_BIO"])
            pred_tokens = pred.iloc[j]["_tokens"]
            pred_tags = pred.iloc[j]["_tags"]

            if len(eval_tokens) != len(gold_tags):
                raise AssertionError(f"Manifest token/tag mismatch at row {j}")
            if gold.iloc[idx]["_tokens"][: len(eval_tokens)] != eval_tokens:
                raise AssertionError(f"Manifest/gold token mismatch at original row {idx}")
            if gold.iloc[idx]["_tags"][: len(gold_tags)] != gold_tags:
                raise AssertionError(f"Manifest/gold tag mismatch at original row {idx}")
            if pred_tokens[: len(eval_tokens)] != eval_tokens:
                raise AssertionError(f"Prediction token mismatch at evaluation row {j}")
            if len(pred_tags) < len(eval_tokens):
                raise AssertionError(f"Prediction shorter than evaluated prefix at row {j}")

            y_true.append(gold_tags)
            y_pred.append(pred_tags[: len(eval_tokens)])
    else:
        if len(pred) != len(gold):
            raise ValueError("Prediction and gold row counts differ")
        for i in range(len(gold)):
            if "Review_Text" in gold.columns and "Review_Text" in pred.columns:
                if norm_text(gold.iloc[i]["Review_Text"]) != norm_text(pred.iloc[i]["Review_Text"]):
                    raise AssertionError(f"Review mismatch at row {i}")
            if gold.iloc[i]["_tokens"] != pred.iloc[i]["_tokens"]:
                raise AssertionError(f"Token mismatch at row {i}")
            if len(pred.iloc[i]["_tags"]) != len(gold.iloc[i]["_tags"]):
                raise AssertionError(f"Prediction tag length mismatch at row {i}")
            y_true.append(gold.iloc[i]["_tags"])
            y_pred.append(pred.iloc[i]["_tags"])

    return y_true, y_pred


def metric_tables(y_true: Sequence[Sequence[str]], y_pred: Sequence[Sequence[str]]):
    report = classification_report(
        y_true, y_pred, mode="strict", scheme=IOB2,
        output_dict=True, zero_division=0,
    )
    summary_rows = []
    for metric in ["micro avg", "macro avg", "weighted avg"]:
        r = report[metric]
        summary_rows.append({
            "Metric": metric,
            "Precision": float(r["precision"]),
            "Recall": float(r["recall"]),
            "F1": float(r["f1-score"]),
            "Support": int(r["support"]),
        })
    per_class = []
    for name, r in report.items():
        if name in {"micro avg", "macro avg", "weighted avg"} or not isinstance(r, dict):
            continue
        if "f1-score" in r:
            per_class.append({
                "Category": name,
                "Precision": float(r["precision"]),
                "Recall": float(r["recall"]),
                "F1": float(r["f1-score"]),
                "Support": int(r["support"]),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(per_class), report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--manifest", default=None,
                    help="Optional common evaluation manifest with row indices, evaluated tokens and gold BIO")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--system-name", default="ATE system")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = sequences(args.gold, args.pred, args.manifest)
    summary, per_class, report = metric_tables(y_true, y_pred)
    summary.insert(0, "System", args.system_name)
    if not per_class.empty:
        per_class.insert(0, "System", args.system_name)

    summary.to_csv(out / "strict_span_metrics.csv", index=False)
    per_class.to_csv(out / "strict_span_per_category.csv", index=False)
    (out / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
