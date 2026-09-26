#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train supervised transformer baselines on a manifest-defined split.

The script has no fixed corpus-size, support, annotation-budget or target-score
assumptions. Split membership and training hyperparameters are external inputs.
All evaluated systems use a common original-word prefix that fits the configured
maximum sequence length for every selected tokenizer.
"""
from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import random
import shutil
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_SPECS = {
    "mbert": "bert-base-multilingual-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr": "xlm-roberta-base",
}
DISPLAY_NAMES = {"mbert": "mBERT", "indicbert": "IndicBERT", "xlmr": "XLM-R"}


def parse_obj(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    return ast.literal_eval(str(x))


def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in ["Review_Text", "Word_Tokens", "BIO_Tags"]:
        if col not in df.columns:
            raise KeyError(f"Gold CSV missing column: {col}")
    df["tokens"] = df["Word_Tokens"].apply(lambda x: [str(t) for t in parse_obj(x)])
    df["pairs"] = df["BIO_Tags"].apply(parse_obj)
    df["tags"] = df["pairs"].apply(lambda x: [str(tag) for _, tag in x])
    for i, row in df.iterrows():
        pair_tokens = [str(tok) for tok, _ in row["pairs"]]
        if pair_tokens != row["tokens"]:
            raise AssertionError(f"Word_Tokens/BIO token mismatch at row {i}")
    return df


def apply_filters(df: pd.DataFrame, filters: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for item in filters:
        if "=" not in item:
            raise ValueError(f"Invalid manifest filter: {item!r}; expected COLUMN=VALUE")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if key not in out.columns:
            raise KeyError(f"Manifest filter column not found: {key}")
        numeric = pd.to_numeric(out[key], errors="coerce")
        try:
            value_num = float(raw)
            mask = numeric == value_num
            if mask.any():
                out = out[mask]
                continue
        except ValueError:
            pass
        out = out[out[key].astype(str) == raw]
    return out.copy()


def load_split_indices(path: str, filters: Sequence[str], train_role: str, val_role: str):
    m = apply_filters(pd.read_csv(path), filters)
    needed = {"Original_Row_Index", "Role"}
    missing = needed - set(m.columns)
    if missing:
        raise KeyError(f"Split manifest missing columns: {sorted(missing)}")
    train = m.loc[m["Role"].astype(str) == train_role, "Original_Row_Index"].astype(int).tolist()
    val = m.loc[m["Role"].astype(str) == val_role, "Original_Row_Index"].astype(int).tolist()
    if not train or not val:
        raise ValueError("Manifest filter/role selection produced an empty training or validation split")
    if set(train) & set(val):
        raise AssertionError("Training and validation indices overlap")
    return train, val


def load_test_manifest(path: str):
    t = pd.read_csv(path).copy()
    required = {"Original_Row_Index", "Word_Tokens_Evaluated", "Gold_BIO"}
    missing = required - set(t.columns)
    if missing:
        raise KeyError(f"Test manifest missing columns: {sorted(missing)}")
    idx = t["Original_Row_Index"].astype(int).tolist()
    tokens = [[str(x) for x in parse_obj(v)] for v in t["Word_Tokens_Evaluated"]]
    tags = [[str(x) for x in parse_obj(v)] for v in t["Gold_BIO"]]
    return t, idx, tokens, tags


def encoded_length(tokenizer, words: Sequence[str]) -> int:
    if not words:
        return tokenizer.num_special_tokens_to_add(pair=False)
    if getattr(tokenizer, "is_fast", False):
        return len(tokenizer(list(words), is_split_into_words=True, add_special_tokens=True,
                             truncation=False)["input_ids"])
    ids = []
    for word in words:
        ids.extend(tokenizer.encode(str(word), add_special_tokens=False))
    return len(tokenizer.build_inputs_with_special_tokens(ids))


def common_word_limit(words, tokenizers: Dict[str, object], max_len: int) -> int:
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if all(encoded_length(tok, words[:mid]) <= max_len for tok in tokenizers.values()):
            lo = mid
        else:
            hi = mid - 1
    return lo


def add_shared_prefix(df: pd.DataFrame, model_keys: Sequence[str], max_len: int) -> pd.DataFrame:
    tokenizers = {
        k: AutoTokenizer.from_pretrained(MODEL_SPECS[k], use_fast=(k != "indicbert"))
        for k in model_keys
    }
    limits = [common_word_limit(words, tokenizers, max_len) for words in df["tokens"]]
    out = df.copy()
    out["tokens_shared"] = [x[:n] for x, n in zip(out["tokens"], limits)]
    out["tags_shared"] = [x[:n] for x, n in zip(out["tags"], limits)]
    if any(len(x) == 0 for x in out["tokens_shared"]):
        raise AssertionError("A review became empty after common truncation")
    return out


def build_label_maps(df: pd.DataFrame):
    labels = sorted({tag for seq in df["tags"] for tag in seq})
    return labels, {x: i for i, x in enumerate(labels)}, {i: x for i, x in enumerate(labels)}


def make_hf_split(df, indices):
    part = pd.DataFrame({
        "tokens_shared": [df.iloc[int(i)]["tokens_shared"] for i in indices],
        "tags_shared": [df.iloc[int(i)]["tags_shared"] for i in indices],
    })
    return Dataset.from_pandas(part, preserve_index=False)


def make_encode_label_ids(label2id):
    def fn(batch):
        return {"label_ids": [[label2id[t] for t in seq] for seq in batch["tags_shared"]]}
    return fn


def make_tokenize_and_align(tokenizer, max_len: int):
    def fn(example):
        words, labels = example["tokens_shared"], example["label_ids"]
        if getattr(tokenizer, "is_fast", False):
            enc = tokenizer(words, truncation=True, max_length=max_len, is_split_into_words=True)
            aligned, prev = [], None
            for word_idx in enc.word_ids():
                if word_idx is None:
                    aligned.append(-100)
                elif word_idx != prev:
                    aligned.append(labels[word_idx])
                else:
                    aligned.append(-100)
                prev = word_idx
            enc["labels"] = aligned
            return enc

        input_ids, aligned = [], []
        available = max_len - tokenizer.num_special_tokens_to_add(pair=False)
        for word, label in zip(words, labels):
            pieces = tokenizer.encode(str(word), add_special_tokens=False)
            if not pieces:
                continue
            if len(input_ids) + len(pieces) > available:
                break
            input_ids.extend(pieces)
            aligned.extend([label] + [-100] * (len(pieces) - 1))
        final_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        special = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=False)
        final_labels, j = [], 0
        for is_special in special:
            if is_special:
                final_labels.append(-100)
            else:
                final_labels.append(aligned[j]); j += 1
        return {"input_ids": final_ids, "attention_mask": [1] * len(final_ids), "labels": final_labels}
    return fn


def decode_word_level(pred_ids, label_ids, id2label):
    preds_all, gold_all = [], []
    for preds, labels in zip(pred_ids, label_ids):
        pp, gg = [], []
        for p, g in zip(preds, labels):
            if int(g) != -100:
                pp.append(id2label[int(p)]); gg.append(id2label[int(g)])
        preds_all.append(pp); gold_all.append(gg)
    return preds_all, gold_all


def summary_rows(name, y_true, y_pred, n_train, n_val):
    rep = classification_report(y_true, y_pred, mode="strict", scheme=IOB2,
                                output_dict=True, zero_division=0)
    rows = []
    for metric in ["micro avg", "macro avg", "weighted avg"]:
        r = rep[metric]
        rows.append({
            "Model": name,
            "Supervised_Train_Reviews": len(y_true) if False else int(n_train),
            "Supervised_Validation_Reviews": int(n_val),
            "Metric": metric,
            "Precision": float(r["precision"]),
            "Recall": float(r["recall"]),
            "F1": float(r["f1-score"]),
            "Support": int(r["support"]),
        })
    return rows


def make_training_args(output_dir: str, cfg: dict, seed: int):
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        learning_rate=float(cfg["learning_rate"]),
        num_train_epochs=int(cfg["max_epochs"]),
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=str(cfg.get("selection_metric", "eval_loss")),
        greater_is_better=bool(cfg.get("greater_is_better", False)),
        save_total_limit=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=bool(cfg.get("fp16", torch.cuda.is_available()) and torch.cuda.is_available()),
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "epoch"
    if "save_only_model" in params:
        kwargs["save_only_model"] = True
    return TrainingArguments(**kwargs)


def run_one(df, model_key, train_idx, val_idx, test_idx, expected_tokens, expected_gold,
            cfg, label2id, id2label, output_dir: Path, seed: int):
    set_all_seeds(seed)
    name = DISPLAY_NAMES[model_key]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SPECS[model_key], use_fast=(model_key != "indicbert"))
    encode_ids = make_encode_label_ids(label2id)
    tokenize = make_tokenize_and_align(tokenizer, int(cfg["max_length"]))

    def prep(indices):
        ds = make_hf_split(df, indices)
        ds = ds.map(encode_ids, batched=True, load_from_cache_file=False)
        return ds.map(tokenize, batched=False,
                      remove_columns=["tokens_shared", "tags_shared", "label_ids"],
                      load_from_cache_file=False)

    train_ds, val_ds, test_ds = prep(train_idx), prep(val_idx), prep(test_idx)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_SPECS[model_key], num_labels=len(label2id), id2label=id2label,
        label2id=label2id, ignore_mismatched_sizes=True,
    )
    run_dir = output_dir / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.get("checkpoint_root", str(output_dir / "_checkpoints"))) / model_key
    if ckpt.exists():
        shutil.rmtree(ckpt)
    ckpt.mkdir(parents=True, exist_ok=True)

    trainer_kwargs = dict(
        model=model,
        args=make_training_args(str(ckpt), cfg, seed),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(cfg["patience"]))],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    pred_output = trainer.predict(test_ds)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    y_pred, y_true = decode_word_level(pred_ids, pred_output.label_ids, id2label)
    if y_true != expected_gold:
        raise AssertionError(f"{name}: decoded gold differs from common evaluation gold")

    rows = summary_rows(name, y_true, y_pred, len(train_idx), len(val_idx))
    pd.DataFrame(rows).to_csv(run_dir / "strict_span_metrics.csv", index=False)
    pd.DataFrame({
        "Original_Row_Index": test_idx,
        "Review_Text": [df.iloc[int(i)]["Review_Text"] for i in test_idx],
        "Word_Tokens_Evaluated": expected_tokens,
        "Gold_BIO": y_true,
        "Predicted_BIO": y_pred,
    }).to_csv(run_dir / "word_level_predictions.csv", index=False)

    shutil.rmtree(ckpt, ignore_errors=True)
    del trainer, model, tokenizer, train_ds, val_ds, test_ds, pred_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--manifest-filter", action="append", default=[])
    ap.add_argument("--train-role", default="supervised_train")
    ap.add_argument("--validation-role", default="supervised_validation")
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--config", required=True, help="JSON with seed, models and training hyperparameters")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["seed"])
    model_keys = list(cfg["models"])
    unknown = set(model_keys) - set(MODEL_SPECS)
    if unknown:
        raise ValueError(f"Unknown transformer baseline(s): {sorted(unknown)}")

    train_idx, val_idx = load_split_indices(
        args.split_manifest, args.manifest_filter, args.train_role, args.validation_role
    )
    test_manifest, test_idx, manifest_tokens, manifest_gold = load_test_manifest(args.test_manifest)
    if (set(train_idx) | set(val_idx)) & set(test_idx):
        raise AssertionError("Development and evaluation indices overlap")

    df = add_shared_prefix(load_gold(args.gold), model_keys, int(cfg["max_length"]))
    expected_tokens = [df.iloc[int(i)]["tokens_shared"] for i in test_idx]
    expected_gold = [df.iloc[int(i)]["tags_shared"] for i in test_idx]
    if expected_tokens != manifest_tokens or expected_gold != manifest_gold:
        raise AssertionError("Provided test manifest does not match the configured common truncation rule")

    _, label2id, id2label = build_label_maps(df)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    all_rows = []
    for offset, key in enumerate(model_keys):
        all_rows.extend(run_one(
            df, key, train_idx, val_idx, test_idx, expected_tokens, expected_gold,
            cfg, label2id, id2label, out, seed + offset,
        ))
    pd.DataFrame(all_rows).to_csv(out / "baseline_metrics.csv", index=False)
    print(pd.DataFrame(all_rows).to_string(index=False))


if __name__ == "__main__":
    main()
