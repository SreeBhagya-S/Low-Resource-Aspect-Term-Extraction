#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate ATE BIO tags from a versioned experiment configuration.

This wrapper keeps tuned/selected hyperparameters outside source code. The JSON
configuration is copied to the output directory together with SHA-256 hashes of
resource files, making the run auditable without embedding observed scores or
fixed dataset sizes in the implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from ate_pipeline_core import main as pipeline_main


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(cfg: Dict[str, Any], dotted: str):
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing configuration key: {dotted}")
        cur = cur[part]
    return cur


def append_many(argv: List[str], flag: str, values: Iterable[str]) -> None:
    for value in values:
        argv.extend([flag, str(value)])


def build_pipeline_argv(args, cfg: Dict[str, Any]) -> List[str]:
    model = require(cfg, "model")
    thresholds = require(cfg, "thresholds")
    components = cfg.get("components", {})
    semantic = cfg.get("semantic_discovery", {})
    neighbor = cfg.get("neighbor_extension", {})

    argv = [
        "--input", args.input,
        "--output", args.output,
        "--text-column", cfg.get("text_column", "Review_Text"),
        "--lexicon", args.lexicon,
        "--stopwords", args.stopwords,
        "--model", str(model["name"]),
        "--max-length", str(model["max_length"]),
        "--fuzzy-cutoff", str(require(cfg, "fuzzy_cutoff")),
        "--exact-context-threshold", str(thresholds["exact"]),
        "--trigger-context-threshold", str(thresholds["trigger"]),
        "--fuzzy-context-threshold", str(thresholds["fuzzy"]),
        "--category-margin-threshold", str(require(cfg, "category_margin_threshold")),
        "--taxonomy", str(cfg.get("taxonomy", "evaluation16")),
    ]

    if args.device:
        argv.extend(["--device", args.device])

    append_many(argv, "--suffix-file", args.suffix_file)
    append_many(argv, "--sentiment-file", args.sentiment_file)

    # Optional discovery/extension modules are enabled only by configuration.
    if semantic.get("enabled", False):
        argv.append("--enable-semantic-discovery")
        argv.extend(["--semantic-score-threshold", str(semantic["score_threshold"])])
        argv.extend(["--semantic-margin-threshold", str(semantic["margin_threshold"])])

    if neighbor.get("enabled", False):
        argv.append("--enable-neighbor-extension")
        argv.extend(["--adjacent-context-threshold", str(neighbor["threshold"])])

    # Active components are represented positively in the JSON config. The
    # core generator exposes disable switches for controlled ablation.
    if not components.get("morphology", True):
        argv.append("--disable-morphology")
    if not components.get("stopword_sentiment_filter", True):
        argv.append("--disable-stopword-sentiment")
    if not components.get("fuzzy_matching", True):
        argv.append("--disable-fuzzy")
    if not components.get("regex_matching", True):
        argv.append("--disable-regex")
    if not components.get("contextual_filtering", True):
        argv.append("--no-context")
    if not components.get("category_guard", True):
        argv.append("--disable-category-guard")
    if not components.get("evidence_overlap_resolution", True):
        argv.append("--disable-evidence-overlap")

    if cfg.get("allow_wordnet_fallback", False):
        argv.append("--allow-wordnet-fallback")

    if args.diagnostics:
        argv.extend(["--diagnostics", args.diagnostics])

    return argv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="JSON pipeline configuration")
    ap.add_argument("--input", required=True, help="Input review CSV")
    ap.add_argument("--output", required=True, help="Generated BIO-tag CSV")
    ap.add_argument("--lexicon", required=True)
    ap.add_argument("--suffix-file", action="append", required=True)
    ap.add_argument("--stopwords", required=True)
    ap.add_argument("--sentiment-file", action="append", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--diagnostics", default=None)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    argv = build_pipeline_argv(args, cfg)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    resources = [args.config, args.lexicon, args.stopwords, *args.suffix_file, *args.sentiment_file]
    run_record = {
        "configuration": cfg,
        "input": str(Path(args.input)),
        "output": str(output),
        "resources": {str(Path(p)): sha256(p) for p in resources},
    }
    (output.parent / f"{output.stem}.run.json").write_text(
        json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pipeline_main(argv)


if __name__ == "__main__":
    main()
