#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_dual_supervision_baselines.py
=================================
Leakage-free two-regime supervised baseline experiment for Manglish ATE.

Protocol
--------
The 1,000 manually annotated reviews are split ONCE with outer seed 42:
  * 800-review development partition
  * 200-review final held-out test partition

Within the 800-review development partition, the same seed-42 split defines:
  * 720 reviews for full-supervision training
  * 80 reviews for full-supervision validation / pipeline configuration pool

Two supervised baseline regimes are then run against the SAME 200-review test:

1. FULL SUPERVISION (primary supervised reference)
   * 720 training reviews
   * 80 validation reviews
   * 200 held-out test reviews

2. LOW-RESOURCE (secondary, pre-specified resource study)
   * the exact same 80-review validation/configuration pool is treated as the
     total manual-label budget
   * it is split once into 64 training + 16 validation reviews
   * the same 200 held-out test reviews are used only for final scoring

The low-resource experiment is not selected as a replacement for the primary
full-supervision experiment. Both are saved and should be reported together.
No observed score is hard-coded or used to choose a regime.

All transformer baselines use one shared original-word prefix that fits within
MAX_LEN for mBERT, IndicBERT and XLM-R. Evaluation uses strict span-level IOB2
metrics via seqeval. Validation data, never test data, are used for checkpoint
selection and early stopping.
"""
from __future__ import annotations

import argparse
import ast
import gc
import inspect
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, f1_score as seq_f1
from seqeval.scheme import IOB2

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

MAX_LEN = 128
MODEL_SPECS = {
    "mbert": "bert-base-multilingual-cased",
    "indicbert": "ai4bharat/indic-bert",
    "xlmr": "xlm-roberta-base",
}
DISPLAY_NAMES = {
    "mbert": "mBERT",
    "indicbert": "IndicBERT",
    "xlmr": "XLM-R",
    "bilstm": "BiLSTM+CRF",
}


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


def fixed_outer_split(n: int, seed: int):
    all_idx = np.arange(n)
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=0.20, random_state=seed, shuffle=True
    )
    train720_idx, val80_idx = train_test_split(
        dev_idx, test_size=0.10, random_state=seed, shuffle=True
    )
    if n == 1000:
        assert len(train720_idx) == 720 and len(val80_idx) == 80 and len(test_idx) == 200
    return np.asarray(train720_idx), np.asarray(val80_idx), np.asarray(test_idx)


def load_gold(path: str):
    df = pd.read_csv(path)
    for col in ["Review_Text", "Word_Tokens", "BIO_Tags"]:
        if col not in df.columns:
            raise KeyError(f"Gold CSV missing column: {col}")
    if len(df) != 1000:
        raise ValueError(f"Expected 1000 reviews, found {len(df)}")
    df = df.copy()
    df["tokens"] = df["Word_Tokens"].apply(lambda x: [str(t) for t in parse_obj(x)])
    df["pairs"] = df["BIO_Tags"].apply(parse_obj)
    df["tags"] = df["pairs"].apply(lambda x: [str(tag) for _, tag in x])
    for i, row in df.iterrows():
        pair_tokens = [str(tok) for tok, _ in row["pairs"]]
        if pair_tokens != row["tokens"]:
            raise AssertionError(f"Word_Tokens/BIO token mismatch at row {i}")
    return df


def encoded_length(tokenizer, words: Sequence[str]) -> int:
    if len(words) == 0:
        return tokenizer.num_special_tokens_to_add(pair=False)
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(
            list(words), is_split_into_words=True, add_special_tokens=True,
            truncation=False,
        )
        return len(enc["input_ids"])
    ids = []
    for word in words:
        ids.extend(tokenizer.encode(str(word), add_special_tokens=False))
    return len(tokenizer.build_inputs_with_special_tokens(ids))


def common_word_limit(words: Sequence[str], tokenizers: Dict[str, object], max_len: int = MAX_LEN) -> int:
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        ok = all(encoded_length(tok, words[:mid]) <= max_len for tok in tokenizers.values())
        if ok:
            lo = mid
        else:
            hi = mid - 1
    return lo


def add_shared_prefix(df: pd.DataFrame):
    print("Loading three tokenizers to establish one common word-level truncation rule...")
    budget_tokenizers = {
        "mbert": AutoTokenizer.from_pretrained(MODEL_SPECS["mbert"], use_fast=True),
        "indicbert": AutoTokenizer.from_pretrained(MODEL_SPECS["indicbert"], use_fast=False),
        "xlmr": AutoTokenizer.from_pretrained(MODEL_SPECS["xlmr"], use_fast=True),
    }
    limits = []
    for i, words in enumerate(df["tokens"], start=1):
        limits.append(common_word_limit(words, budget_tokenizers, MAX_LEN))
        if i % 100 == 0 or i == len(df):
            print(f"  truncation limits {i}/{len(df)}")
    df = df.copy()
    df["shared_word_limit"] = limits
    df["tokens_shared"] = [x[:n] for x, n in zip(df["tokens"], limits)]
    df["tags_shared"] = [x[:n] for x, n in zip(df["tags"], limits)]
    if any(len(x) == 0 for x in df["tokens_shared"]):
        raise AssertionError("A review became empty after shared truncation")
    return df


def span_metric_rows(model: str, budget: int, repeat: int, y_true, y_pred, n_train: int, n_val: int):
    rep = classification_report(
        y_true, y_pred, mode="strict", scheme=IOB2,
        output_dict=True, zero_division=0,
    )
    rows = []
    for metric in ["micro avg", "macro avg", "weighted avg"]:
        r = rep[metric]
        rows.append({
            "Model": model,
            "Total_Label_Budget": int(budget),
            "Repeat": int(repeat),
            "Supervised_Train_Reviews": int(n_train),
            "Supervised_Validation_Reviews": int(n_val),
            "Metric": metric,
            "Precision": float(r["precision"]),
            "Recall": float(r["recall"]),
            "F1": float(r["f1-score"]),
            "Support": int(r["support"]),
        })
    return rows


def make_budget_pool(train720_idx, val80_idx, budget: int, repeat_seed: int):
    """Nested-style pool anchored on the exact 80 reviews used by the proposed pipeline.

    * B <= 80: deterministic subset of the fixed 80 validation reviews.
    * B > 80: all fixed 80 validation reviews + B-80 reviews from the 720 train reviews.
    """
    if budget < 4 or budget > 800:
        raise ValueError("Each budget must be between 4 and 800")
    rng = np.random.default_rng(repeat_seed)
    val_perm = rng.permutation(val80_idx)
    train_perm = rng.permutation(train720_idx)
    if budget <= 80:
        pool = val_perm[:budget]
    else:
        pool = np.concatenate([val80_idx, train_perm[:budget - 80]])
    return np.asarray(pool, dtype=int)


def split_budget_pool(pool: np.ndarray, seed: int, val_fraction: float):
    if len(pool) < 4:
        raise ValueError("Budget too small for an internal train/validation split")
    n_val = max(1, int(round(len(pool) * val_fraction)))
    n_val = min(n_val, len(pool) - 1)
    train_idx, val_idx = train_test_split(
        pool, test_size=n_val, random_state=seed, shuffle=True
    )
    return np.asarray(train_idx), np.asarray(val_idx)


def build_label_maps(df: pd.DataFrame):
    labels = sorted({tag for seq in df["tags"] for tag in seq})
    label2id = {x: i for i, x in enumerate(labels)}
    id2label = {i: x for x, i in label2id.items()}
    return labels, label2id, id2label


def make_hf_split(df: pd.DataFrame, indices: Sequence[int]):
    part = pd.DataFrame({
        "tokens_shared": [df.iloc[int(i)]["tokens_shared"] for i in indices],
        "tags_shared": [df.iloc[int(i)]["tags_shared"] for i in indices],
    })
    return Dataset.from_pandas(part, preserve_index=False)


def make_encode_label_ids(label2id):
    def fn(batch):
        return {"label_ids": [[label2id[t] for t in seq] for seq in batch["tags_shared"]]}
    return fn


def make_tokenize_and_align(tokenizer):
    def fn(example):
        words = example["tokens_shared"]
        original_labels = example["label_ids"]
        if getattr(tokenizer, "is_fast", False):
            tokenized = tokenizer(
                words, truncation=True, max_length=MAX_LEN,
                is_split_into_words=True,
            )
            word_ids = tokenized.word_ids()
            aligned = []
            previous_word = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned.append(-100)
                elif word_idx != previous_word:
                    aligned.append(original_labels[word_idx])
                else:
                    aligned.append(-100)
                previous_word = word_idx
            tokenized["labels"] = aligned
            return tokenized

        input_ids = []
        aligned_labels = []
        available = MAX_LEN - tokenizer.num_special_tokens_to_add(pair=False)
        for word, label in zip(words, original_labels):
            pieces = tokenizer.encode(str(word), add_special_tokens=False)
            if not pieces:
                continue
            if len(input_ids) + len(pieces) > available:
                break
            input_ids.extend(pieces)
            aligned_labels.append(label)
            aligned_labels.extend([-100] * (len(pieces) - 1))

        final_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=False)
        final_labels = []
        j = 0
        for is_special in special_mask:
            if is_special:
                final_labels.append(-100)
            else:
                final_labels.append(aligned_labels[j])
                j += 1
        return {
            "input_ids": final_ids,
            "attention_mask": [1] * len(final_ids),
            "labels": final_labels,
        }
    return fn


def decode_word_level(pred_ids, label_ids, id2label):
    preds_all, gold_all = [], []
    for preds, labels in zip(pred_ids, label_ids):
        pp, gg = [], []
        for p, g in zip(preds, labels):
            if int(g) != -100:
                pp.append(id2label[int(p)])
                gg.append(id2label[int(g)])
        preds_all.append(pp)
        gold_all.append(gg)
    return preds_all, gold_all


def training_args(output_dir: str, seed: int, batch_size: int, max_epochs: int, patience: int):
    kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-5,
        num_train_epochs=max_epochs,
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        report_to="none",
        seed=seed,
        data_seed=seed,
        fp16=bool(torch.cuda.is_available()),
    )
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    # Avoid writing large optimizer/scheduler states. Only the model checkpoint
    # is needed to reload the best epoch for final prediction.
    if "save_only_model" in params:
        kwargs["save_only_model"] = True
    return TrainingArguments(**kwargs)


def run_transformer(df, train_idx, val_idx, test_idx, expected_test_gold,
                    model_key, budget, repeat, seed, label2id, id2label,
                    out_dir: Path, batch_size: int, max_epochs: int, patience: int):
    model_name = MODEL_SPECS[model_key]
    display = DISPLAY_NAMES[model_key]
    print(f"\n{'='*88}\n{display} | budget={budget} | repeat={repeat}\n{'='*88}")
    set_all_seeds(seed)

    use_fast = False if model_key == "indicbert" else True
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    tokenize_fn = make_tokenize_and_align(tok)
    encode_ids = make_encode_label_ids(label2id)

    def prep(indices):
        ds = make_hf_split(df, indices)
        ds = ds.map(encode_ids, batched=True, load_from_cache_file=False)
        ds = ds.map(
            tokenize_fn, batched=False,
            remove_columns=["tokens_shared", "tags_shared", "label_ids"],
            load_from_cache_file=False,
        )
        return ds

    train_ds, val_ds, test_ds = prep(train_idx), prep(val_idx), prep(test_idx)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    # Final lightweight results go to Drive/output_dir, but Trainer checkpoints
    # are written to local Colab storage. Saving multi-GB optimizer checkpoints
    # directly to a mounted Google Drive can raise torch iostream errors.
    run_dir = out_dir / f"budget_{budget:04d}" / f"repeat_{repeat:02d}" / model_key
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path("/content/ate_budget_checkpoints")
        / f"budget_{budget:04d}"
        / f"repeat_{repeat:02d}"
        / model_key
    )
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    args = training_args(str(checkpoint_dir), seed, batch_size, max_epochs, patience)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForTokenClassification(tok),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tok
    else:
        trainer_kwargs["tokenizer"] = tok
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    print("Best checkpoint:", trainer.state.best_model_checkpoint)

    pred_output = trainer.predict(test_ds)
    pred_ids = np.argmax(pred_output.predictions, axis=-1)
    y_pred, y_true = decode_word_level(pred_ids, pred_output.label_ids, id2label)
    if y_true != expected_test_gold:
        raise AssertionError(f"{display}: decoded gold does not match common test gold")

    rows = span_metric_rows(display, budget, repeat, y_true, y_pred, len(train_idx), len(val_idx))
    pd.DataFrame(rows).to_csv(run_dir / "strict_span_metrics.csv", index=False)
    pred_df = pd.DataFrame({
        "Original_Row_Index": list(map(int, test_idx)),
        "Review_Text": [df.iloc[int(i)]["Review_Text"] for i in test_idx],
        "Word_Tokens_Evaluated": [df.iloc[int(i)]["tokens_shared"] for i in test_idx],
        "Gold_BIO": y_true,
        "Predicted_BIO": y_pred,
    })
    pred_df.to_csv(run_dir / "word_level_predictions.csv", index=False)

    print(pd.DataFrame(rows).to_string(index=False))
    del trainer, model, tok, train_ds, val_ds, test_ds, pred_output
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # Free local temporary checkpoints after metrics/predictions are saved.
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir, ignore_errors=True)
    return rows


def run_bilstm(df, train_idx, val_idx, test_idx, expected_test_gold,
               budget, repeat, seed, label2id, id2label, out_dir: Path,
               epochs: int = 5):
    try:
        from torchcrf import CRF
    except ImportError as e:
        raise RuntimeError(
            "BiLSTM+CRF requested but torchcrf is missing. Install with: pip install pytorch-crf"
        ) from e
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    display = DISPLAY_NAMES["bilstm"]
    print(f"\n{'='*88}\n{display} | budget={budget} | repeat={repeat}\n{'='*88}")
    set_all_seeds(seed)

    PAD, UNK = "<PAD>", "<UNK>"
    word2id = {PAD: 0, UNK: 1}
    for i in train_idx:
        for w in df.iloc[int(i)]["tokens_shared"]:
            key = str(w).casefold()
            if key not in word2id:
                word2id[key] = len(word2id)
    pad_id, unk_id = 0, 1
    o_id = label2id["O"]

    class WordDataset(TorchDataset):
        def __init__(self, indices):
            self.examples = []
            for idx in indices:
                words = df.iloc[int(idx)]["tokens_shared"]
                tags = df.iloc[int(idx)]["tags_shared"]
                x = [word2id.get(str(w).casefold(), unk_id) for w in words]
                y = [label2id[t] for t in tags]
                self.examples.append((int(idx), x, y))
        def __len__(self): return len(self.examples)
        def __getitem__(self, i):
            idx, x, y = self.examples[i]
            return {"row_idx": idx,
                    "input_ids": torch.tensor(x, dtype=torch.long),
                    "labels": torch.tensor(y, dtype=torch.long)}

    def collate(batch):
        xs = [b["input_ids"] for b in batch]
        ys = [b["labels"] for b in batch]
        rows = [b["row_idx"] for b in batch]
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        xpad = pad_sequence(xs, batch_first=True, padding_value=pad_id)
        ypad = pad_sequence(ys, batch_first=True, padding_value=o_id)
        mask = torch.arange(xpad.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
        return {"row_idx": rows, "input_ids": xpad, "labels": ypad,
                "mask": mask.bool(), "lengths": lengths}

    class BiLSTMCRF(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(word2id), 128, padding_idx=pad_id)
            self.lstm = nn.LSTM(128, 128, num_layers=1, bidirectional=True, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.classifier = nn.Linear(256, len(label2id))
            self.crf = CRF(len(label2id), batch_first=True)
        def emissions(self, x, mask):
            x = self.embedding(x)
            lengths = mask.long().sum(dim=1).detach().cpu()
            packed = pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=mask.size(1)
            )
            return self.classifier(self.dropout(x))
        def loss(self, x, y, mask):
            return -self.crf(self.emissions(x, mask), y, mask=mask, reduction="mean")
        def decode(self, x, mask):
            return self.crf.decode(self.emissions(x, mask), mask=mask)

    train_loader = DataLoader(WordDataset(train_idx), batch_size=16, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(WordDataset(val_idx), batch_size=16, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(WordDataset(test_idx), batch_size=16, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMCRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def evaluate(loader):
        model.eval(); rows=[]; gold=[]; pred=[]
        with torch.no_grad():
            for batch in loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                mask = batch["mask"].to(device)
                decoded = model.decode(x, mask)
                for ps, gs, n, ridx in zip(decoded, y, batch["lengths"], batch["row_idx"]):
                    nn_ = int(n.item())
                    pred.append([id2label[int(z)] for z in ps[:nn_]])
                    gold.append([id2label[int(z)] for z in gs[:nn_].tolist()])
                    rows.append(int(ridx))
        return rows, gold, pred

    # Select the BiLSTM+CRF checkpoint using ONLY the supervised validation split.
    # This prevents a low-resource run from collapsing at a later epoch and then
    # evaluating that collapsed final state on the held-out test set.
    # The test set is never used for checkpoint selection.
    import copy
    history = []
    best_val_f1 = -1.0
    best_epoch = None
    best_state = None

    no_improve = 0
    bilstm_patience = 7

    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            mask = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = model.loss(x, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step(); losses.append(float(loss.item()))

        _, vg, vp = evaluate(val_loader)
        vf1 = seq_f1(vg, vp, mode="strict", scheme=IOB2, zero_division=0)
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_micro_f1": float(vf1),
        })
        print(f"Epoch {epoch}/{epochs} loss={np.mean(losses):.4f} val_micro_f1={vf1:.4f}")

        if float(vf1) > best_val_f1 + 1e-12:
            best_val_f1 = float(vf1)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= bilstm_patience:
            print(
                f"BiLSTM+CRF early stopping after {epoch} epochs; "
                f"best validation epoch={best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("No BiLSTM+CRF validation checkpoint was selected")

    model.load_state_dict(best_state)
    print(f"Restored BiLSTM+CRF best validation checkpoint: epoch={best_epoch}, val_micro_f1={best_val_f1:.4f}")

    test_order, y_true, y_pred = evaluate(test_loader)
    if test_order != list(map(int, test_idx)):
        raise AssertionError("BiLSTM test row order mismatch")
    if y_true != expected_test_gold:
        raise AssertionError("BiLSTM gold differs from common test gold")

    run_dir = out_dir / f"budget_{budget:04d}" / f"repeat_{repeat:02d}" / "bilstm"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    rows = span_metric_rows(display, budget, repeat, y_true, y_pred, len(train_idx), len(val_idx))
    pd.DataFrame(rows).to_csv(run_dir / "strict_span_metrics.csv", index=False)
    pd.DataFrame({
        "Original_Row_Index": list(map(int, test_idx)),
        "Review_Text": [df.iloc[int(i)]["Review_Text"] for i in test_idx],
        "Word_Tokens_Evaluated": [df.iloc[int(i)]["tokens_shared"] for i in test_idx],
        "Gold_BIO": y_true,
        "Predicted_BIO": y_pred,
    }).to_csv(run_dir / "word_level_predictions.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    del model
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True,
                    help="Aligned 1,000-review gold CSV with Review_Text, Word_Tokens, BIO_Tags")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--test-manifest", required=True,
                    help="Locked previous evaluation_manifest.csv with the exact 200 held-out test rows")
    ap.add_argument("--low-resource-manifest", required=True,
                    help="Locked previous split_manifest.csv with the exact 64 train / 16 validation rows")
    ap.add_argument("--models", nargs="+", choices=["mbert", "indicbert", "xlmr", "bilstm"],
                    default=["mbert", "indicbert", "xlmr", "bilstm"])
    ap.add_argument("--outer-seed", type=int, default=42,
                    help="Frozen seed for the 800-development / 200-test split")
    ap.add_argument("--low-resource-seed", type=int, default=3002,
                    help="Base training-seed schedule for the locked low-resource run; "
                         "3002 reproduces the earlier model seeds 3003/3004/3005/3006")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--early-stopping-patience", type=int, default=5)
    ap.add_argument("--bilstm-epochs", type=int, default=5)
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-low-resource", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    df = load_gold(args.gold)
    df = add_shared_prefix(df)
    labels, label2id, id2label = build_label_maps(df)

    # ---------------------------------------------------------------- locked outer split
    # Reconstruct the seed-42 outer/full split, then verify that the supplied
    # PREVIOUS test manifest is exactly the same 200-review test set.
    train720_idx, val80_idx, generated_test_idx = fixed_outer_split(len(df), args.outer_seed)
    dev800_idx = np.concatenate([train720_idx, val80_idx])

    assert len(dev800_idx) == 800
    assert len(train720_idx) == 720
    assert len(val80_idx) == 80
    assert len(generated_test_idx) == 200
    assert not (set(dev800_idx) & set(generated_test_idx))
    assert not (set(train720_idx) & set(val80_idx))

    supplied_test = pd.read_csv(args.test_manifest).copy()
    required_test_cols = {"Original_Row_Index", "Word_Tokens_Evaluated", "Gold_BIO"}
    missing_test = required_test_cols - set(supplied_test.columns)
    if missing_test:
        raise KeyError(f"Locked test manifest missing columns: {sorted(missing_test)}")
    if len(supplied_test) != 200 or supplied_test["Original_Row_Index"].nunique() != 200:
        raise AssertionError("Locked test manifest must contain exactly 200 unique reviews")

    test_idx = supplied_test["Original_Row_Index"].astype(int).to_numpy()
    if list(map(int, test_idx)) != list(map(int, generated_test_idx)):
        raise AssertionError(
            "The supplied previous evaluation_manifest.csv is not exactly the "
            "seed-42 fixed outer test set in the same row order. Stop rather than "
            "silently changing the final test set."
        )

    # Verify that the previously saved evaluated token prefixes/gold labels still
    # match the current common-prefix construction. This protects against silent
    # tokenizer/version drift.
    expected_test_gold = []
    for row_i, row in supplied_test.iterrows():
        idx = int(row["Original_Row_Index"])
        old_tokens = [str(x) for x in parse_obj(row["Word_Tokens_Evaluated"])]
        old_gold = [str(x) for x in parse_obj(row["Gold_BIO"])]
        now_tokens = [str(x) for x in df.iloc[idx]["tokens_shared"]]
        now_gold = [str(x) for x in df.iloc[idx]["tags_shared"]]
        if old_tokens != now_tokens or old_gold != now_gold:
            raise AssertionError(
                f"Locked test manifest differs from current common evaluation prefix "
                f"at manifest row {row_i}, original row {idx}. "
                "Do not compare results until the environment/tokenization is reconciled."
            )
        expected_test_gold.append(old_gold)

    # Preserve the exact previous test manifest in the final result directory.
    test_manifest = supplied_test.copy()
    test_manifest_path = outdir / "fixed_test_manifest_seed42.csv"
    test_manifest.to_csv(test_manifest_path, index=False)

    self_report = classification_report(
        expected_test_gold, expected_test_gold,
        mode="strict", scheme=IOB2, output_dict=True, zero_division=0,
    )
    support = int(self_report["micro avg"]["support"])

    # ------------------------------------------------ locked low-resource split
    # IMPORTANT: use the exact PREVIOUS 64/16 assignment.  Do not generate a new
    # train/validation subdivision after observing results.
    locked_low = pd.read_csv(args.low_resource_manifest).copy()
    required_low_cols = {"Original_Row_Index", "Role"}
    missing_low = required_low_cols - set(locked_low.columns)
    if missing_low:
        raise KeyError(f"Locked low-resource manifest missing columns: {sorted(missing_low)}")

    if "Total_Label_Budget" in locked_low.columns:
        locked_low = locked_low[locked_low["Total_Label_Budget"].astype(int) == 80].copy()
    if "Repeat" in locked_low.columns:
        locked_low = locked_low[locked_low["Repeat"].astype(int) == 0].copy()

    manifest_train_idx = locked_low.loc[
        locked_low["Role"].astype(str) == "supervised_train", "Original_Row_Index"
    ].astype(int).to_numpy()
    manifest_val_idx = locked_low.loc[
        locked_low["Role"].astype(str) == "supervised_validation", "Original_Row_Index"
    ].astype(int).to_numpy()

    if len(manifest_train_idx) != 64 or len(set(map(int, manifest_train_idx))) != 64:
        raise AssertionError(f"Expected exactly 64 unique previous low-resource train rows; got {len(manifest_train_idx)}")
    if len(manifest_val_idx) != 16 or len(set(map(int, manifest_val_idx))) != 16:
        raise AssertionError(f"Expected exactly 16 unique previous low-resource validation rows; got {len(manifest_val_idx)}")
    if set(map(int, manifest_train_idx)) & set(map(int, manifest_val_idx)):
        raise AssertionError("Previous low-resource train/validation rows overlap")

    low_pool = set(map(int, manifest_train_idx)) | set(map(int, manifest_val_idx))
    val80_set_check = set(map(int, val80_idx))
    if low_pool != val80_set_check:
        only_previous = sorted(low_pool - val80_set_check)
        only_current = sorted(val80_set_check - low_pool)
        raise AssertionError(
            "The previous 80-review low-resource pool is not exactly the fixed "
            "seed-42 validation/configuration pool. "
            f"Only in previous={only_previous[:10]}, only in current={only_current[:10]}"
        )
    if low_pool & set(map(int, test_idx)):
        raise AssertionError("Previous low-resource rows overlap the final test set")

    # --------------------------------------------------------------------------
    # EXACT PREVIOUS LOW-RESOURCE ORDER
    #
    # The old budget runner did NOT train in the row order written to
    # split_manifest.csv. It first permuted val80_idx with repeat_seed=3002 and
    # then train_test_split(..., random_state=3019). The manifest preserves
    # membership but loses that returned array order. SGD is order-sensitive.
    #
    # Reconstruct the exact arrays used by the previous run, then verify their
    # MEMBERSHIP against split_manifest.csv. No score is used here.
    # --------------------------------------------------------------------------
    previous_repeat_seed = args.outer_seed + 37 * 80  # 42 + 2960 = 3002
    if int(args.low_resource_seed) != int(previous_repeat_seed):
        raise AssertionError(
            f"For exact previous-run reproduction, --low-resource-seed must be "
            f"{previous_repeat_seed}; got {args.low_resource_seed}"
        )

    previous_pool = make_budget_pool(
        train720_idx, val80_idx, 80, previous_repeat_seed
    )
    reconstructed_train_idx, reconstructed_val_idx = split_budget_pool(
        previous_pool, previous_repeat_seed + 17, val_fraction=0.20
    )

    if set(map(int, reconstructed_train_idx)) != set(map(int, manifest_train_idx)):
        raise AssertionError(
            "Reconstructed previous 64-review training membership does not match "
            "the supplied split_manifest.csv."
        )
    if set(map(int, reconstructed_val_idx)) != set(map(int, manifest_val_idx)):
        raise AssertionError(
            "Reconstructed previous 16-review validation membership does not match "
            "the supplied split_manifest.csv."
        )

    # Use the reconstructed ORIGINAL ARRAY ORDER for training/validation.
    low_train_idx = np.asarray(reconstructed_train_idx, dtype=int)
    low_val_idx = np.asarray(reconstructed_val_idx, dtype=int)

    print("Locked low-resource membership: 64 train / 16 validation")
    print("Exact previous training order reconstructed: YES")
    print("Previous repeat seed:", previous_repeat_seed)
    print("Previous split seed:", previous_repeat_seed + 17)
    print("First 10 reconstructed train rows:", low_train_idx[:10].tolist())
    print("Validation rows in reconstructed order:", low_val_idx.tolist())

    print("LOCKED SPLIT CHECK PASSED:")
    print("  previous test manifest = exact seed-42 200-review test")
    print("  previous low-resource split = exact 64 train / 16 validation")
    print("  previous 80-review pool = exact seed-42 validation/configuration pool")

    # Save a single row-level protocol manifest.
    rows = []
    train720_set = set(map(int, train720_idx))
    val80_set = set(map(int, val80_idx))
    test_set = set(map(int, test_idx))
    low_train_set = set(map(int, low_train_idx))
    low_val_set = set(map(int, low_val_idx))

    for i in range(len(df)):
        if i in test_set:
            outer = "final_test"
        else:
            outer = "development"
        if i in train720_set:
            full_role = "train"
        elif i in val80_set:
            full_role = "validation"
        else:
            full_role = "test"
        if i in low_train_set:
            low_role = "train"
        elif i in low_val_set:
            low_role = "validation"
        elif i in test_set:
            low_role = "test"
        else:
            low_role = "not_used"
        pipeline_role = "configuration" if i in val80_set else ("test" if i in test_set else "not_used")
        rows.append({
            "Original_Row_Index": i,
            "Outer_Role": outer,
            "Full_Supervised_Role": full_role,
            "Low_Resource_Role": low_role,
            "Pipeline_Role": pipeline_role,
        })

    protocol_manifest = pd.DataFrame(rows)
    protocol_manifest.to_csv(outdir / "dual_regime_protocol_manifest.csv", index=False)

    protocol = {
        "gold_file": str(Path(args.gold).resolve()),
        "outer_seed": args.outer_seed,
        "development_reviews": 800,
        "final_test_reviews": 200,
        "full_supervision": {"train": 720, "validation": 80},
        "pipeline_configuration_reviews": 80,
        "low_resource": {
            "total_label_budget": 80,
            "train": 64,
            "validation": 16,
            "split_source": str(Path(args.low_resource_manifest).resolve()),
            "split_definition": "exact previous 64/16 assignment; no new low-resource split generated",
            "pool_definition": "exact fixed 80-review validation/configuration pool",
            "ordering_rule": "reconstructed from original budget runner: repeat_seed=3002 then split_seed=3019; split_manifest verifies membership",
            "model_seed_base": args.low_resource_seed,
            "model_seed_schedule": {
                "mbert": args.low_resource_seed + 1,
                "indicbert": args.low_resource_seed + 2,
                "xlmr": args.low_resource_seed + 3,
                "bilstm": args.low_resource_seed + 4,
            },
        },
        "test_gold_support": support,
        "test_manifest_source": str(Path(args.test_manifest).resolve()),
        "test_manifest": str(test_manifest_path),
        "selection_rule": "Both full-supervision and low-resource regimes are reported; low-resource is not selected based on test performance.",
    }
    (outdir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print("FROZEN TWO-REGIME PROTOCOL")
    print("=" * 100)
    print(json.dumps(protocol, indent=2))

    all_rows = []

    # ------------------------------------------------------- full supervision
    if not args.skip_full:
        print("\n" + "=" * 100)
        print("PRIMARY SUPERVISED REFERENCE: 720 TRAIN / 80 VALIDATION / 200 TEST")
        print("=" * 100)
        for model_key in args.models:
            # Match the original corrected full-supervision design: fixed seed 42.
            run_seed = args.outer_seed
            if model_key == "bilstm":
                metric_rows = run_bilstm(
                    df, train720_idx, val80_idx, test_idx, expected_test_gold,
                    800, 0, run_seed, label2id, id2label, outdir / "full_supervision",
                    epochs=args.bilstm_epochs,
                )
            else:
                metric_rows = run_transformer(
                    df, train720_idx, val80_idx, test_idx, expected_test_gold,
                    model_key, 800, 0, run_seed, label2id, id2label,
                    outdir / "full_supervision", args.batch_size, args.max_epochs,
                    args.early_stopping_patience,
                )
            for r in metric_rows:
                r["Experiment"] = "Full supervision"
            all_rows.extend(metric_rows)

    # --------------------------------------------------------- low resource 80
    if not args.skip_low_resource:
        print("\n" + "=" * 100)
        print("SECONDARY LOW-RESOURCE STUDY: 64 TRAIN / 16 VALIDATION / SAME 200 TEST")
        print("=" * 100)
        # Preserve the SAME deterministic model-seed schedule used by the earlier
        # budget-80 execution: base 3002 -> 3003/3004/3005/3006.
        for model_key in args.models:
            run_seed = args.low_resource_seed + {"mbert": 1, "indicbert": 2, "xlmr": 3, "bilstm": 4}[model_key]
            if model_key == "bilstm":
                metric_rows = run_bilstm(
                    df, low_train_idx, low_val_idx, test_idx, expected_test_gold,
                    80, 0, run_seed, label2id, id2label, outdir / "low_resource",
                    epochs=args.bilstm_epochs,
                )
            else:
                metric_rows = run_transformer(
                    df, low_train_idx, low_val_idx, test_idx, expected_test_gold,
                    model_key, 80, 0, run_seed, label2id, id2label,
                    outdir / "low_resource", args.batch_size, args.max_epochs,
                    args.early_stopping_patience,
                )
            for r in metric_rows:
                r["Experiment"] = "Low resource (80 total labels)"
            all_rows.extend(metric_rows)

    metrics = pd.DataFrame(all_rows)
    metrics_path = outdir / "DUAL_REGIME_ALL_METRICS.csv"
    metrics.to_csv(metrics_path, index=False)

    if not metrics.empty:
        micro = metrics[metrics["Metric"] == "micro avg"].copy()
        cols = [
            "Experiment", "Model", "Supervised_Train_Reviews",
            "Supervised_Validation_Reviews", "Precision", "Recall", "F1", "Support"
        ]
        micro = micro[cols].sort_values(["Experiment", "F1"], ascending=[True, False])
        micro.to_csv(outdir / "DUAL_REGIME_MICRO_TABLE.csv", index=False)
        print("\n" + "=" * 100)
        print("MEASURED MICRO RESULTS — REPORT BOTH REGIMES")
        print("=" * 100)
        print(micro.to_string(index=False))

    print("\nSaved:", metrics_path)
    print("Saved:", test_manifest_path)
    print("Saved:", outdir / "dual_regime_protocol_manifest.csv")
    print("Saved:", outdir / "protocol.json")
    print("\nInterpretation rule: the full-supervision comparison is the primary supervised reference. The low-resource comparison is a separate resource-sensitivity analysis and must not be substituted for the primary comparison because its ordering is more favorable.")


if __name__ == "__main__":
    main()
