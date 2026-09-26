#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Character-aware BiLSTM+CRF ATE baseline on a manifest-defined split.

The implementation addresses lexical sparsity through a character encoder and
uses strict validation-based checkpoint selection. Hyperparameters and the
random seed are supplied externally in JSON; no observed evaluation score or
fixed corpus size is embedded in the source.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from seqeval.metrics import classification_report, f1_score as seq_f1
from seqeval.scheme import IOB2
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

try:
    from torchcrf import CRF
except ImportError as exc:
    raise RuntimeError("Install pytorch-crf before running this baseline") from exc

PAD_WORD, UNK_WORD = "<PAD>", "<UNK>"
PAD_CHAR, UNK_CHAR = "<PAD>", "<UNK>"
DISPLAY_NAME = "Char-BiLSTM+CRF"


def parse_obj(x):
    if isinstance(x, (list, tuple, dict)):
        return x
    return ast.literal_eval(str(x))


def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gold(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in ["Review_Text", "Word_Tokens", "BIO_Tags"]:
        if col not in df.columns:
            raise KeyError(f"Gold CSV missing column: {col}")
    df["tokens"] = df["Word_Tokens"].apply(lambda x: [str(t) for t in parse_obj(x)])
    df["pairs"] = df["BIO_Tags"].apply(parse_obj)
    df["tags"] = df["pairs"].apply(lambda x: [str(tag) for _, tag in x])
    for i, row in df.iterrows():
        if [str(t) for t, _ in row["pairs"]] != row["tokens"]:
            raise AssertionError(f"Gold token/BIO mismatch at row {i}")
    return df


def apply_filters(df: pd.DataFrame, filters: Sequence[str]) -> pd.DataFrame:
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


def load_split(path, filters, train_role, val_role):
    m = apply_filters(pd.read_csv(path), filters)
    if not {"Original_Row_Index", "Role"}.issubset(m.columns):
        raise KeyError("Split manifest requires Original_Row_Index and Role")
    train = m.loc[m["Role"].astype(str) == train_role, "Original_Row_Index"].astype(int).tolist()
    val = m.loc[m["Role"].astype(str) == val_role, "Original_Row_Index"].astype(int).tolist()
    if not train or not val:
        raise ValueError("Training or validation split is empty")
    if set(train) & set(val):
        raise AssertionError("Training and validation overlap")
    return train, val


def load_test(path, gold):
    t = pd.read_csv(path).copy()
    required = {"Original_Row_Index", "Word_Tokens_Evaluated", "Gold_BIO"}
    if not required.issubset(t.columns):
        raise KeyError(f"Test manifest requires {sorted(required)}")
    idx = t["Original_Row_Index"].astype(int).tolist()
    tokens = [[str(x) for x in parse_obj(v)] for v in t["Word_Tokens_Evaluated"]]
    tags = [[str(x) for x in parse_obj(v)] for v in t["Gold_BIO"]]
    for j, (i, tok, lab) in enumerate(zip(idx, tokens, tags)):
        if tok != gold.iloc[i]["tokens"][:len(tok)] or lab != gold.iloc[i]["tags"][:len(lab)]:
            raise AssertionError(f"Test manifest/gold mismatch at row {j}")
    return idx, tokens, tags


def build_maps(gold):
    labels = sorted({x for seq in gold["tags"] for x in seq})
    if "O" in labels:
        labels = ["O"] + [x for x in labels if x != "O"]
    return labels, {x:i for i,x in enumerate(labels)}, {i:x for i,x in enumerate(labels)}


def build_vocabs(gold, train_idx):
    word2id = {PAD_WORD: 0, UNK_WORD: 1}
    char2id = {PAD_CHAR: 0, UNK_CHAR: 1}
    for idx in train_idx:
        for word in gold.iloc[idx]["tokens"]:
            w = str(word).casefold()
            if w not in word2id:
                word2id[w] = len(word2id)
            for ch in str(word):
                if ch not in char2id:
                    char2id[ch] = len(char2id)
    return word2id, char2id


class SeqDataset(Dataset):
    def __init__(self, row_idx, tokens, tags, word2id, char2id, label2id):
        self.items = []
        uw, uc = word2id[UNK_WORD], char2id[UNK_CHAR]
        for i, words, labs in zip(row_idx, tokens, tags):
            self.items.append({
                "row_idx": int(i), "tokens": list(words),
                "word_ids": [word2id.get(str(w).casefold(), uw) for w in words],
                "char_ids": [[char2id.get(ch, uc) for ch in str(w)] or [uc] for w in words],
                "labels": [label2id[str(x)] for x in labs],
            })
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def make_collate(o_id: int):
    def collate(batch):
        B = len(batch); lengths = torch.tensor([len(x["word_ids"]) for x in batch], dtype=torch.long)
        W = int(lengths.max()); C = max(len(c) for x in batch for c in x["char_ids"])
        word = torch.zeros(B, W, dtype=torch.long)
        chars = torch.zeros(B, W, C, dtype=torch.long)
        char_lengths = torch.ones(B, W, dtype=torch.long)
        labels = torch.full((B, W), o_id, dtype=torch.long)
        mask = torch.zeros(B, W, dtype=torch.bool)
        rows, token_lists = [], []
        for b, x in enumerate(batch):
            n = len(x["word_ids"]); rows.append(x["row_idx"]); token_lists.append(x["tokens"])
            word[b,:n] = torch.tensor(x["word_ids"])
            labels[b,:n] = torch.tensor(x["labels"])
            mask[b,:n] = True
            for j,c in enumerate(x["char_ids"]):
                char_lengths[b,j] = len(c); chars[b,j,:len(c)] = torch.tensor(c)
        return {"word_ids":word,"chars":chars,"char_lengths":char_lengths,"lengths":lengths,
                "labels":labels,"mask":mask,"row_indices":rows,"tokens":token_lists}
    return collate


class CharBiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_labels, cfg):
        super().__init__()
        we = int(cfg["word_embedding_dim"]); ce = int(cfg["char_embedding_dim"])
        ch = int(cfg["char_hidden_dim"]); wh = int(cfg["word_hidden_dim"])
        self.word_embedding = nn.Embedding(n_words, we, padding_idx=0)
        self.char_embedding = nn.Embedding(n_chars, ce, padding_idx=0)
        self.char_lstm = nn.LSTM(ce, ch, batch_first=True, bidirectional=True)
        self.word_lstm = nn.LSTM(we + 2*ch, wh, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(float(cfg["dropout"]))
        self.classifier = nn.Linear(2*wh, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def char_encode(self, chars, char_lengths):
        B,W,C = chars.shape
        flat = chars.reshape(B*W,C); lens = char_lengths.reshape(B*W).clamp(min=1)
        emb = self.char_embedding(flat)
        packed = pack_padded_sequence(emb, lens.cpu(), batch_first=True, enforce_sorted=False)
        _,(h,_) = self.char_lstm(packed)
        return torch.cat([h[-2],h[-1]],dim=-1).reshape(B,W,-1)

    def emissions(self, batch):
        w = self.word_embedding(batch["word_ids"])
        c = self.char_encode(batch["chars"], batch["char_lengths"])
        x = self.dropout(torch.cat([w,c],dim=-1))
        packed = pack_padded_sequence(x, batch["lengths"].cpu(), batch_first=True, enforce_sorted=False)
        packed_out,_ = self.word_lstm(packed)
        out,_ = pad_packed_sequence(packed_out,batch_first=True,total_length=batch["word_ids"].size(1))
        return self.classifier(self.dropout(out))

    def loss(self, batch, class_weights, aux_weight):
        emissions = self.emissions(batch)
        crf_loss = -self.crf(emissions, batch["labels"], mask=batch["mask"], reduction="mean")
        active = batch["mask"]
        ce = nn.functional.cross_entropy(emissions[active], batch["labels"][active], weight=class_weights)
        return crf_loss + aux_weight * ce


def legal(prev, cur):
    if cur.startswith("I-"):
        cat = cur[2:]
        return prev in {f"B-{cat}", f"I-{cat}"}
    return True


def constrained_decode(model, emissions, length, id2label, observed_ids):
    emissions = emissions[:length].detach().clone(); n = emissions.size(-1)
    for j in range(n):
        if j not in observed_ids: emissions[:,j] = -1e9
    start = model.crf.start_transitions.detach().clone(); trans = model.crf.transitions.detach().clone()
    end = model.crf.end_transitions.detach().clone()
    for j in range(n):
        if id2label[j].startswith("I-"): start[j] = -1e9
    for i in range(n):
        for j in range(n):
            if not legal(id2label[i], id2label[j]): trans[i,j] = -1e9
    score = start + emissions[0]; backs=[]
    for t in range(1,length):
        cand = score.unsqueeze(1)+trans; best,bp = cand.max(dim=0); score=best+emissions[t]; backs.append(bp)
    last=int(torch.argmax(score+end).item()); path=[last]
    for bp in reversed(backs): last=int(bp[last].item()); path.append(last)
    return list(reversed(path))


def move(batch, device):
    for k in ["word_ids","chars","char_lengths","lengths","labels","mask"]:
        batch[k]=batch[k].to(device)
    return batch


def evaluate(model, loader, device, id2label, observed_ids):
    model.eval(); yt,yp,records=[],[],[]
    with torch.no_grad():
        for batch in loader:
            batch=move(batch,device); emissions=model.emissions(batch)
            for b in range(emissions.size(0)):
                n=int(batch["lengths"][b].item())
                gold_ids=batch["labels"][b,:n].cpu().tolist()
                pred_ids=constrained_decode(model, emissions[b], n, id2label, observed_ids)
                g=[id2label[x] for x in gold_ids]; p=[id2label[x] for x in pred_ids]
                yt.append(g); yp.append(p)
                records.append({"Original_Row_Index":batch["row_indices"][b],"Word_Tokens_Evaluated":batch["tokens"][b],
                                "Gold_BIO":g,"Predicted_BIO":p})
    return seq_f1(yt,yp,mode="strict",scheme=IOB2),yt,yp,records


def class_weights(gold, train_idx, label2id, cfg, device):
    counts=Counter(x for idx in train_idx for x in gold.iloc[idx]["tags"])
    arr=np.array([counts.get(label,0) for label in label2id],dtype=np.float32)
    w=np.zeros_like(arr); nz=arr>0
    w[nz]=arr[nz].sum()/(nz.sum()*arr[nz]); w=np.sqrt(w)
    w[label2id["O"]] *= float(cfg["outside_class_weight_scale"])
    return torch.tensor(w,dtype=torch.float32,device=device)


def metric_rows(y_true,y_pred,n_train,n_val):
    rep=classification_report(y_true,y_pred,mode="strict",scheme=IOB2,output_dict=True,zero_division=0)
    rows=[]
    for m in ["micro avg","macro avg","weighted avg"]:
        r=rep[m]; rows.append({"Model":DISPLAY_NAME,"Supervised_Train_Reviews":n_train,
            "Supervised_Validation_Reviews":n_val,"Metric":m,"Precision":float(r["precision"]),
            "Recall":float(r["recall"]),"F1":float(r["f1-score"]),"Support":int(r["support"])})
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold",required=True); ap.add_argument("--split-manifest",required=True)
    ap.add_argument("--manifest-filter",action="append",default=[])
    ap.add_argument("--train-role",default="supervised_train"); ap.add_argument("--validation-role",default="supervised_validation")
    ap.add_argument("--test-manifest",required=True); ap.add_argument("--config",required=True); ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()

    cfg=json.loads(Path(args.config).read_text(encoding="utf-8")); seed=int(cfg["seed"]); set_all_seeds(seed)
    device=torch.device(cfg.get("device","cuda" if torch.cuda.is_available() else "cpu"))
    gold=load_gold(args.gold); train_idx,val_idx=load_split(args.split_manifest,args.manifest_filter,args.train_role,args.validation_role)
    test_idx,test_tokens,test_tags=load_test(args.test_manifest,gold)
    if (set(train_idx)|set(val_idx)) & set(test_idx): raise AssertionError("Development/evaluation overlap")
    labels,label2id,id2label=build_maps(gold); word2id,char2id=build_vocabs(gold,train_idx)
    observed={label2id[x] for idx in train_idx for x in gold.iloc[idx]["tags"]}; observed.add(label2id["O"])

    train_ds=SeqDataset(train_idx,[gold.iloc[i]["tokens"] for i in train_idx],[gold.iloc[i]["tags"] for i in train_idx],word2id,char2id,label2id)
    val_ds=SeqDataset(val_idx,[gold.iloc[i]["tokens"] for i in val_idx],[gold.iloc[i]["tags"] for i in val_idx],word2id,char2id,label2id)
    test_ds=SeqDataset(test_idx,test_tokens,test_tags,word2id,char2id,label2id)
    collate=make_collate(label2id["O"]); gen=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_ds,batch_size=int(cfg["batch_size"]),shuffle=True,generator=gen,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=int(cfg["batch_size"]),shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=int(cfg["batch_size"]),shuffle=False,collate_fn=collate)

    model=CharBiLSTMCRF(len(word2id),len(char2id),len(labels),cfg).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(cfg["learning_rate"]))
    weights=class_weights(gold,train_idx,label2id,cfg,device); aux=float(cfg["auxiliary_ce_weight"])
    best_f1=-1.0; best_state=None; best_epoch=None; patience=0; history=[]
    for epoch in range(1,int(cfg["max_epochs"])+1):
        model.train(); total=0.0
        for batch in train_loader:
            batch=move(batch,device); opt.zero_grad(); loss=model.loss(batch,weights,aux); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float(cfg["gradient_clip_norm"])); opt.step(); total+=float(loss.item())
        vf1,_,_,_=evaluate(model,val_loader,device,id2label,observed)
        history.append({"epoch":epoch,"train_loss":total/max(1,len(train_loader)),"validation_micro_f1":vf1})
        if vf1 > best_f1 + float(cfg.get("improvement_tolerance",1e-8)):
            best_f1=vf1; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; patience=0
        else:
            patience+=1
        if patience >= int(cfg["patience"]): break
    if best_state is None: raise RuntimeError("No validation checkpoint was created")
    model.load_state_dict(best_state)
    _,yt,yp,records=evaluate(model,test_loader,device,id2label,observed)

    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=metric_rows(yt,yp,len(train_idx),len(val_idx)); pd.DataFrame(rows).to_csv(out/"strict_span_metrics.csv",index=False)
    pd.DataFrame(records).to_csv(out/"word_level_predictions.csv",index=False)
    pd.DataFrame(history).to_csv(out/"training_history.csv",index=False)
    (out/"run_config.json").write_text(json.dumps({**cfg,"selected_epoch":best_epoch},indent=2),encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=="__main__": main()
