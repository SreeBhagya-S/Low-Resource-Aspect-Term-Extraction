#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate the reviewer-facing ATE experiment from one JSON project file.

The workflow records commands and relies on external manifests/configuration
files for data splits and hyperparameters. No observed result is used to choose
or modify a later stage.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(map(str, cmd)))
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.run([str(x) for x in cmd], stdout=log, stderr=subprocess.STDOUT)
    if p.returncode:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        print("\n".join(tail))
        raise RuntimeError(f"Stage failed; see {log_path}")


def add_resources(cmd, r):
    cmd += ["--lexicon", r["lexicon"], "--stopwords", r["stopwords"]]
    for p in r["suffix_files"]: cmd += ["--suffix-file", p]
    for p in r["sentiment_files"]: cmd += ["--sentiment-file", p]
    if r.get("device"): cmd += ["--device", r["device"]]


def add_manifest_filters(cmd, filters):
    for item in filters or []:
        cmd += ["--manifest-filter", item]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True, help="Reviewer project JSON")
    args = ap.parse_args()

    project_path = Path(args.project).resolve()
    cfg = json.loads(project_path.read_text(encoding="utf-8"))
    root = Path(cfg["output_root"]); root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    logs = root / "logs"

    paths = cfg["paths"]; resources = cfg["resources"]
    split_filters = cfg.get("split_filters", [])

    # 1. Development-only pipeline configuration selection (optional if a
    #    previously frozen configuration is supplied).
    selected_config = paths.get("pipeline_config")
    if cfg.get("run_configuration_selection", False):
        select_out = root / "configuration_selection"
        cmd = [sys.executable, scripts / "select_pipeline_config.py",
               "--generator", scripts / "generate_ate_bio.py",
               "--search-space", paths["pipeline_search_space"],
               "--input", paths["input"], "--gold", paths["gold"],
               "--split-manifest", paths["split_manifest"],
               "--validation-role", cfg.get("validation_role", "supervised_validation"),
               "--output-dir", select_out]
        add_manifest_filters(cmd, split_filters); add_resources(cmd, resources)
        run(cmd, logs / "01_configuration_selection.log")
        selected_config = str(select_out / "selected_pipeline_config.json")
    if not selected_config:
        raise ValueError("Provide paths.pipeline_config or enable configuration selection")

    # 2. Generate frozen pipeline predictions for the complete input.
    pipeline_out = root / "pipeline"; pipeline_out.mkdir(exist_ok=True)
    pipeline_pred = pipeline_out / "pipeline_predictions.csv"
    cmd = [sys.executable, scripts / "generate_ate_bio.py",
           "--config", selected_config, "--input", paths["input"],
           "--output", pipeline_pred, "--diagnostics", pipeline_out / "candidate_diagnostics.csv"]
    add_resources(cmd, resources)
    run(cmd, logs / "02_generate_pipeline.log")

    # 3. Transformer baselines.
    transformer_out = root / "baselines" / "transformers"
    cmd = [sys.executable, scripts / "train_transformer_baselines.py",
           "--gold", paths["gold"], "--split-manifest", paths["split_manifest"],
           "--test-manifest", paths["test_manifest"], "--config", paths["transformer_config"],
           "--train-role", cfg.get("train_role", "supervised_train"),
           "--validation-role", cfg.get("validation_role", "supervised_validation"),
           "--output-dir", transformer_out]
    add_manifest_filters(cmd, split_filters)
    run(cmd, logs / "03_transformer_baselines.log")

    # 4. Optional recurrent baseline.
    recurrent_pred = None
    if paths.get("char_bilstm_config"):
        recurrent_out = root / "baselines" / "char_bilstm_crf"
        cmd = [sys.executable, scripts / "train_char_bilstm_crf.py",
               "--gold", paths["gold"], "--split-manifest", paths["split_manifest"],
               "--test-manifest", paths["test_manifest"], "--config", paths["char_bilstm_config"],
               "--train-role", cfg.get("train_role", "supervised_train"),
               "--validation-role", cfg.get("validation_role", "supervised_validation"),
               "--output-dir", recurrent_out]
        add_manifest_filters(cmd, split_filters)
        run(cmd, logs / "04_char_bilstm_crf.log")
        recurrent_pred = recurrent_out / "word_level_predictions.csv"

    # 5. Recompute all comparison metrics from prediction files.
    compare_out = root / "comparison"
    cmd = [sys.executable, scripts / "compare_methods.py",
           "--gold", paths["gold"], "--manifest", paths["test_manifest"],
           "--system", f"Proposed weakly supervised ATE={pipeline_pred}"]
    transformer_cfg = json.loads(Path(paths["transformer_config"]).read_text(encoding="utf-8"))
    display = {"mbert":"mBERT","indicbert":"IndicBERT","xlmr":"XLM-R"}
    for key in transformer_cfg["models"]:
        cmd += ["--system", f"{display.get(key,key)}={transformer_out / key / 'word_level_predictions.csv'}"]
    if recurrent_pred is not None:
        cmd += ["--system", f"Char-BiLSTM+CRF={recurrent_pred}"]
    cmd += ["--output-dir", compare_out]
    run(cmd, logs / "05_compare_methods.log")

    # 6. Component ablation of the same frozen pipeline on the common manifest.
    ablation_out = root / "ablation"
    cmd = [sys.executable, scripts / "ablation_study.py",
           "--generator", scripts / "generate_ate_bio.py", "--config", selected_config,
           "--input", paths["input"], "--gold", paths["gold"], "--manifest", paths["test_manifest"],
           "--output-dir", ablation_out]
    add_resources(cmd, resources)
    run(cmd, logs / "06_ablation.log")

    print("Workflow completed. Output root:", root)


if __name__ == "__main__":
    main()
