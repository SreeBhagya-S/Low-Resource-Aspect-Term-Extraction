#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core weakly supervised aspect-term extraction pipeline for Malayalam-English
code-mixed reviews.

The generator is independent of gold labels. Experimental hyperparameters are
provided through command-line arguments (or a wrapper configuration) and are
not selected from evaluation outputs inside this module.

Output schema
-------------
Review_Text, Word_Tokens, Updated_Aspect_Terms, BIO_Tags
"""

import argparse
import ast
import difflib
import itertools
import math
import os
import re
import string
import subprocess
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

def _ensure_deps() -> None:
    """Auto-install missing pip packages required by the methodology pipeline."""
    missing = []
    for pkg, import_name in [("nltk", "nltk"), ("transformers", "transformers")]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Installing missing dependencies: {', '.join(missing)} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            stdout=subprocess.DEVNULL,
        )

def _ensure_wordnet() -> None:
    """Auto-download NLTK WordNet data if not already present."""
    try:
        import nltk
        from nltk.corpus import wordnet
        # Trigger a resource lookup; raises LookupError if data is missing.
        wordnet.synsets("test")
    except LookupError:
        print("Downloading NLTK WordNet data ...")
        import nltk
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

_ensure_deps()
_ensure_wordnet()

import numpy as np
import pandas as pd

"""## Configuration"""

ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF]")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@[A-Za-z0-9_]+")
SPACE_RE = re.compile(r"\s+")

# Punctuation is removed as a token boundary, but apostrophes/hyphens are kept
# inside lexical units because they matter for forms such as battery's / ui-ux.
EDGE_PUNCT = string.punctuation.replace("'", "").replace("-", "") + "“”‘’…।"

# Category name chosen in the current release.  The supplied lexicon still uses
# the header `screen-`, while the current annotation files use B-display/I-display.
CATEGORY_ALIASES = {
    "screen": "display",
}

# Category inventory used by the manually annotated evaluation corpus, with
# the repository's current `display` naming replacing the manuscript's `screen`.
EVALUATION_CATEGORIES = {
    "phone", "camera", "price", "software", "display", "network", "design",
    "os", "battery", "memory", "applications", "general", "processor",
    "speaker", "hardware", "tablet",
}

# Technical category headers present in the supplied lexicon.  This also stops
# arbitrary regex lines ending in '-' from being mistaken for a header.
KNOWN_CATEGORY_HEADERS = {
    "memory", "screen", "display", "ui-ux", "camera", "phone", "os",
    "tablet", "laptop", "battery", "software", "processor", "applications",
    "price", "speaker", "network", "hardware", "design", "general", "game",
}

# Multiword matching is sufficient for the supplied lexicon; a four-token cap
# also matches the manuscript's examples and the historical extraction code.
MAX_NGRAM = 4

# Explicit morphology examples/rules stated in the manuscript and present in
# the historical ate_bio.py. The supplied suffix resources are supplementary
# and do not contain every possessive marker (notably ന്റെ / യുടെ).
CORE_MAL_SUFFIXES = {
    "യുടെ", "ന്റെ", "നുള്ള", "ത്തിൽ", "കളെ", "ങ്ങൾ", "ങ്ങളായി",
    "മായി", "പ്പെട്ടു", "നിന്റെ",
}

# Context markers used only to prevent bare numerical values from being treated
# as semantic aspects when the supplied regex/seed entries are intentionally broad.
MEMORY_CONTEXT = {"gb", "mb", "kb", "tb", "ജിബി", "ram", "storage", "memory"}

PRICE_CONTEXT = {"₹", "rs", "rs.", "inr", "rupee", "rupees", "price", "rate", "വില",
                 "k", "l", "thousand", "thousnd", "thusnd", "thousandrupees"}

"""## Text/resource utilities"""

def unicode_nfc(text: object) -> str:
    """Unicode NFC normalization plus removal of zero-width characters."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = ZERO_WIDTH_RE.sub("", s)
    return s

def clean_review_text(text: object) -> str:
    """Minimal cleanup: NFC, zero-width removal, URL/mention removal, spaces."""
    s = unicode_nfc(text)
    s = URL_RE.sub(" ", s)
    s = MENTION_RE.sub(" ", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s

def surface_tokenize(text: str) -> List[str]:
    """Tokenize while preserving Malayalam/mixed-script lexical units.

    The output intentionally excludes punctuation because the historical
    ATE_Bio_tags.csv stores lexical tokens rather than punctuation tokens.
    """
    text = clean_review_text(text)
    # Split first at whitespace and strong punctuation, then trim edge symbols.
    chunks = re.split(r"[\s,;:!?()\[\]{}\"“”]+", text)
    out: List[str] = []
    for chunk in chunks:
        tok = chunk.strip().strip(EDGE_PUNCT)
        # Leading/trailing dashes are punctuation (e.g. "---വരും"), while
        # internal hyphens such as ui-ux are preserved.
        tok = tok.strip("-").strip()
        if not tok:
            continue
        # Split repeated full stops but preserve decimal-like forms.
        if "." in tok and not re.fullmatch(r"\d+(?:\.\d+)+", tok):
            parts = [p.strip(EDGE_PUNCT) for p in re.split(r"\.{2,}|(?<!\d)\.(?!\d)", tok)]
        else:
            parts = [tok]
        out.extend(unicode_nfc(p).lower() for p in parts if p)
    return out

def normalize_basic(text: str) -> str:
    s = unicode_nfc(text).lower().replace("’", "'")
    s = s.strip().strip(EDGE_PUNCT)
    s = SPACE_RE.sub(" ", s)
    return s

def contains_malayalam(text: str) -> bool:
    return any("\u0D00" <= ch <= "\u0D7F" for ch in text)

def load_word_set(path: Optional[str]) -> Set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resource not found: {p}")
    words: Set[str] = set()
    for encoding in ("utf-8", "utf-8-sig", "windows-1252"):
        try:
            with p.open("r", encoding=encoding) as f:
                for line in f:
                    line = normalize_basic(line.strip())
                    if line:
                        words.add(line)
            return words
        except UnicodeDecodeError:
            words.clear()
    raise UnicodeError(f"Could not decode {p}")

def load_union(paths: Sequence[str]) -> Set[str]:
    out: Set[str] = set()
    for p in paths:
        out.update(load_word_set(p))
    return out

def looks_like_regex(entry: str) -> bool:
    markers = (r"(?:", r"\d", r"\s", r"\b", r"[", r"(?=", r"(?<")
    return any(m in entry for m in markers)

@dataclass(frozen=True)
class LexiconEntry:
    category: str
    original_category: str
    raw: str
    normalized: str
    tokens: Tuple[str, ...]
    regex: bool = False
    category_trigger: bool = False

def parse_aspect_lexicon(path: str) -> Tuple[Dict[str, List[LexiconEntry]], List[LexiconEntry]]:
    """Parse `category-` sections and map screen -> display for released tags."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Aspect lexicon not found: {p}")

    by_category: Dict[str, List[LexiconEntry]] = defaultdict(list)
    entries: List[LexiconEntry] = []
    current_original: Optional[str] = None
    current_category: Optional[str] = None

    with p.open("r", encoding="utf-8-sig") as f:
        for raw_line in f:
            raw = unicode_nfc(raw_line).strip()
            if not raw:
                continue

            header_candidate = normalize_basic(raw[:-1]) if raw.endswith("-") else ""
            if raw.endswith("-") and header_candidate in KNOWN_CATEGORY_HEADERS:
                current_original = header_candidate
                current_category = CATEGORY_ALIASES.get(header_candidate, header_candidate)
                # Category header itself is also a lexical trigger. For `screen-`
                # this gives the term `screen` but the output category `display`.
                trigger = LexiconEntry(
                    category=current_category,
                    original_category=current_original,
                    raw=header_candidate,
                    normalized=header_candidate,
                    tokens=tuple(surface_tokenize(header_candidate)),
                    regex=False,
                    category_trigger=True,
                )
                by_category[current_category].append(trigger)
                entries.append(trigger)
                continue

            if current_category is None or current_original is None:
                continue

            is_re = looks_like_regex(raw)
            norm = normalize_basic(raw) if not is_re else raw.strip()
            toks = tuple(normalize_basic(t) for t in surface_tokenize(norm)) if not is_re else tuple()
            if not is_re and not toks:
                continue
            entry = LexiconEntry(
                category=current_category,
                original_category=current_original,
                raw=raw,
                normalized=norm,
                tokens=toks,
                regex=is_re,
                category_trigger=False,
            )
            by_category[current_category].append(entry)
            entries.append(entry)

    # Deduplicate exact entries after category aliasing.
    dedup: Dict[Tuple[str, str, bool], LexiconEntry] = {}
    for e in entries:
        key = (e.category, e.normalized, e.regex)
        old = dedup.get(key)
        if old is None or e.category_trigger:
            dedup[key] = e
    entries = list(dedup.values())
    by_category = defaultdict(list)
    for e in entries:
        by_category[e.category].append(e)
    return dict(by_category), entries

"""## Lemmatization / variant generation"""

class Normalizer:
    def __init__(self, suffixes: Set[str]):
        # Longest-first prevents a short suffix from hiding a more specific one.
        self.suffixes = sorted({normalize_basic(s) for s in suffixes if normalize_basic(s)},
                               key=len, reverse=True)
        self._wnl = None
        self.wordnet_available = False
        try:
            from nltk.stem import WordNetLemmatizer
            self._wnl = WordNetLemmatizer()
            # Force a resource lookup once so a missing corpus is detected here.
            _ = self._wnl.lemmatize("screens")
            self.wordnet_available = True
        except Exception:
            self._wnl = None
            self.wordnet_available = False

    def english_lemma(self, token: str) -> str:
        t = normalize_basic(token)
        if t.endswith("'s") and len(t) > 3:
            t = t[:-2]
        if self._wnl is not None:
            try:
                # noun first, then verb; ATE candidates are predominantly nouns.
                noun = self._wnl.lemmatize(t, pos="n")
                return normalize_basic(noun)
            except Exception:
                pass
        # Deterministic fallback used only when NLTK WordNet data is unavailable.
        if t.endswith("ies") and len(t) > 4:
            return t[:-3] + "y"
        if t.endswith("ses") and len(t) > 4:
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss") and len(t) > 3:
            return t[:-1]
        return t

    def suffix_roots(self, token: str) -> Set[str]:
        """Generate plausible Malayalam/mixed-script base forms without over-writing surface text."""
        base = normalize_basic(token)
        roots = {base}
        frontier = {base}
        # At most two suffix-removal passes; all variants are retained and matching
        # decides whether a root is useful. This is safer than destructive stemming.
        for _ in range(2):
            nxt: Set[str] = set()
            for form in frontier:
                for suffix in self.suffixes:
                    if len(form) <= len(suffix) + 2:
                        continue
                    if form.endswith(suffix):
                        stem = normalize_basic(form[:-len(suffix)])
                        if len(stem) >= 2 and stem not in roots:
                            roots.add(stem)
                            nxt.add(stem)
            frontier = nxt
            if not frontier:
                break
        return roots

    def variants(self, token: str) -> Set[str]:
        base = normalize_basic(token)
        if not base:
            return set()
        variants = {base}

        # Malayalam suffixes can be attached to Malayalam or English roots,
        # e.g. ക്യാമറയുടെ and batteryയുടെ.
        if contains_malayalam(base):
            variants.update(self.suffix_roots(base))

        # English lemmatization also helps mixed forms after Malayalam stripping.
        for form in list(variants):
            if not contains_malayalam(form):
                variants.add(self.english_lemma(form))
        return {v for v in variants if v}

"""## Candidate matching"""

class BasicOnlyNormalizer:
    """Ablation normalizer: NFC/lowercase only; no suffix stripping or lemmatization."""
    def variants(self, token: str) -> Set[str]:
        base = normalize_basic(token)
        return {base} if base else set()


@dataclass
class Candidate:
    start: int
    end: int                  # exclusive
    category: str
    canonical: str
    lexical_score: float
    match_type: str           # exact | fuzzy | regex | contextual-extension
    filtered: bool = False
    category_trigger: bool = False
    context_score: Optional[float] = None
    # V4: score of the strongest competing category and target-vs-competitor gap.
    other_category_score: Optional[float] = None
    category_margin: Optional[float] = None

    @property
    def length(self) -> int:
        return self.end - self.start

def max_similarity(variants: Set[str], target: str) -> float:
    if not variants:
        return 0.0
    return max(difflib.SequenceMatcher(None, v, target).ratio() for v in variants)

def best_fuzzy_pair(variants: Set[str], target: str) -> Tuple[str, float]:
    """Return the variant giving the highest SequenceMatcher score."""
    if not variants:
        return "", 0.0
    best = max(variants, key=lambda v: difflib.SequenceMatcher(None, v, target).ratio())
    return best, difflib.SequenceMatcher(None, best, target).ratio()

def fuzzy_pair_allowed(variants: Set[str], target: str, cutoff: float) -> Tuple[bool, float]:
    """Conservative fuzzy gate while respecting the manuscript cutoff >= 0.75.

    A raw 0.75 character-similarity threshold is unsafe for short lexicon items:
    unrelated 4--6 character words can exceed it by chance.  We therefore keep
    0.75 as the minimum threshold, but require >=0.80 for <=5-character pairs,
    >=0.90 for <=3-character pairs, and require the same initial character when
    both strings use the same script.  These are precision guards, not learned
    parameters and do not use gold annotations.
    """
    variant, sim = best_fuzzy_pair(variants, target)
    if not variant or sim < cutoff:
        return False, sim
    m = min(len(variant), len(target))
    if m <= 3 and sim < max(cutoff, 0.90):
        return False, sim
    if m <= 5 and sim < max(cutoff, 0.80):
        return False, sim
    same_script = contains_malayalam(variant) == contains_malayalam(target)
    if same_script and variant and target and variant[0] != target[0]:
        return False, sim
    max_len = max(len(variant), len(target))
    if abs(len(variant) - len(target)) > max(2, int(round(0.40 * max_len))):
        return False, sim
    return True, sim

def is_filtered_surface(token: str, stopwords: Set[str], sentiment: Set[str], normalizer: Normalizer) -> bool:
    variants = normalizer.variants(token)
    return any(v in stopwords or v in sentiment for v in variants)

def has_nearby_context(tokens: Sequence[str], start: int, end: int, markers: Set[str], radius: int = 2) -> bool:
    left = max(0, start - radius)
    right = min(len(tokens), end + radius)
    neighborhood = [normalize_basic(t) for t in tokens[left:right]]
    return any(t in markers for t in neighborhood)

def numerical_candidate_allowed(category: str, tokens: Sequence[str], start: int, end: int) -> bool:
    """Guard against broad numeric seed/regex matches with no domain context."""
    phrase = " ".join(normalize_basic(t) for t in tokens[start:end])
    if not re.fullmatch(r"\d+(?:\.\d+)?", phrase):
        return True
    if category == "phone":
        return False
    if category == "memory":
        return has_nearby_context(tokens, start, end, MEMORY_CONTEXT, radius=1)
    if category == "price":
        return has_nearby_context(tokens, start, end, PRICE_CONTEXT, radius=2)
    return True

class LexiconMatcher:
    """Indexed exact/fuzzy/regex matcher built once for the whole corpus."""

    def __init__(self, entries: Sequence[LexiconEntry]):
        self.exact_index: Dict[Tuple[str, ...], List[LexiconEntry]] = defaultdict(list)
        self.fuzzy_single: Dict[Tuple[bool, str], List[LexiconEntry]] = defaultdict(list)
        self.regex_entries: List[Tuple[LexiconEntry, re.Pattern]] = []

        for e in entries:
            if e.regex:
                try:
                    self.regex_entries.append(
                        (e, re.compile(e.raw, flags=re.IGNORECASE | re.UNICODE))
                    )
                except re.error:
                    continue
                continue

            if not (1 <= len(e.tokens) <= MAX_NGRAM):
                continue
            self.exact_index[e.tokens].append(e)
            # Orthographic fuzzy matching is applied to single lexical units.
            # Multi-word expressions are recovered through normalized exact
            # n-gram matching and contextual extension.
            if len(e.tokens) == 1 and e.tokens[0]:
                term = e.tokens[0]
                self.fuzzy_single[(contains_malayalam(term), term[0])].append(e)

    @staticmethod
    def _variant_combinations(window_vars: Sequence[Set[str]], cap: int = 64):
        lists = [sorted(v) for v in window_vars]
        if any(not x for x in lists):
            return []
        # Suffix/lemmatization normally gives only 1--3 variants. A cap avoids
        # pathological combinatorial expansion on heavily suffixed forms.
        out = []
        for combo in itertools.product(*lists):
            out.append(tuple(combo))
            if len(out) >= cap:
                break
        return out

    def match(
        self,
        tokens: Sequence[str],
        token_variants: Sequence[Set[str]],
        stopwords: Set[str],
        sentiment: Set[str],
        normalizer: Normalizer,
        fuzzy_cutoff: float,
    ) -> List[Candidate]:
        candidates: List[Candidate] = []
        n_tokens = len(tokens)
        seen = set()

        def add_candidate(e: LexiconEntry, start: int, end: int,
                          score: float, match_type: str, canonical: Optional[str] = None):
            key = (start, end, e.category, e.normalized, match_type)
            if key in seen:
                return
            if not numerical_candidate_allowed(e.category, tokens, start, end):
                return
            seen.add(key)
            filtered_flags = [
                is_filtered_surface(tokens[i], stopwords, sentiment, normalizer)
                for i in range(start, end)
            ]
            candidates.append(Candidate(
                start=start,
                end=end,
                category=e.category,
                canonical=canonical if canonical is not None else e.normalized,
                lexical_score=score,
                match_type=match_type,
                filtered=all(filtered_flags),
                category_trigger=e.category_trigger,
            ))

        # 1) Normalized exact n-gram lookup. Longest-first improves both speed
        # and span quality; final overlap handling remains deterministic.
        for n in range(min(MAX_NGRAM, n_tokens), 0, -1):
            for start in range(0, n_tokens - n + 1):
                end = start + n
                for combo in self._variant_combinations(token_variants[start:end]):
                    for e in self.exact_index.get(combo, ()):
                        add_candidate(e, start, end, 1.0, "exact")

        # 2) Fuzzy matching for single normalized lexical units. Bucket by
        # script + first character before SequenceMatcher, which prevents both
        # many false positives and an O(tokens x whole-lexicon) corpus scan.
        for start, variants in enumerate(token_variants):
            checked_entries = set()
            for variant in variants:
                if not variant:
                    continue
                bucket = self.fuzzy_single.get(
                    (contains_malayalam(variant), variant[0]), ()
                )
                for e in bucket:
                    if e in checked_entries:
                        continue
                    checked_entries.add(e)
                    # Exact hits have already been added above.
                    if variant == e.tokens[0]:
                        continue
                    ok, sim = fuzzy_pair_allowed({variant}, e.tokens[0], fuzzy_cutoff)
                    if ok:
                        add_candidate(e, start, start + 1, sim, "fuzzy")

        # 3) Regex triggers, tested against word windows so their matches can
        # be mapped back to exact BIO boundaries.
        for e, pattern in self.regex_entries:
            for n in range(min(MAX_NGRAM, n_tokens), 0, -1):
                for start in range(0, n_tokens - n + 1):
                    end = start + n
                    phrase = " ".join(normalize_basic(t) for t in tokens[start:end])
                    if pattern.fullmatch(phrase):
                        add_candidate(e, start, end, 1.0, "regex", canonical=phrase)

        return candidates

def candidate_matches(
    matcher: LexiconMatcher,
    tokens: Sequence[str],
    token_variants: Sequence[Set[str]],
    stopwords: Set[str],
    sentiment: Set[str],
    normalizer: Normalizer,
    fuzzy_cutoff: float,
) -> List[Candidate]:
    """Compatibility wrapper around the indexed corpus-level matcher."""
    return matcher.match(
        tokens=tokens,
        token_variants=token_variants,
        stopwords=stopwords,
        sentiment=sentiment,
        normalizer=normalizer,
        fuzzy_cutoff=fuzzy_cutoff,
    )

"""## Frozen mBERT contextual encoder"""

class ContextEncoder:
    def __init__(
        self,
        model_name: str,
        by_category: Dict[str, List[LexiconEntry]],
        device: Optional[str] = None,
        max_length: int = 128,
        centroid_batch_size: int = 32,
    ):
        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Context mode requires torch and transformers. "
                "Install with: pip install torch transformers"
            ) from e

        self.torch = torch
        self.F = F
        self.max_length = max_length
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError("A fast tokenizer is required for word_ids() alignment.")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.centroids = self._build_category_centroids(by_category, centroid_batch_size)

    def _mean_pool_texts(self, texts: Sequence[str], batch_size: int) -> List["object"]:
        torch = self.torch
        vectors: List[object] = []
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=32,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special = enc.pop("special_tokens_mask").to(self.device)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                hidden = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].bool() & (~special.bool())
            for row, row_mask in zip(hidden, mask):
                if row_mask.any():
                    vec = row[row_mask].mean(dim=0)
                    vec = self.F.normalize(vec, dim=0)
                    vectors.append(vec)
                else:
                    vectors.append(torch.zeros(hidden.shape[-1], device=self.device))
        return vectors

    def _build_category_centroids(self, by_category, batch_size):
        torch = self.torch
        centroids = {}
        for category, entries in by_category.items():
            # Regex entries cannot be encoded as lexical phrases.
            phrases = []
            seen = set()


            for e in entries:

                # Regex entries cannot be encoded as lexical phrases.
                if e.regex:
                    continue

                # V3:
                # Do not include the category header itself when
                # constructing its semantic centroid.
                # This avoids circular similarity:
                # "phone" should not score highly simply because
                # "phone" was used to construct the phone centroid.
                if e.category_trigger:
                    continue

                phrase = e.normalized

                if len(phrase) < 2 or phrase in seen:
                    continue
                seen.add(phrase)
                phrases.append(phrase)
            if not phrases:
                continue
            vecs = self._mean_pool_texts(phrases, batch_size)
            valid = [v for v in vecs if float(v.norm().item()) > 0]
            if not valid:
                continue
            centroid = torch.stack(valid, dim=0).mean(dim=0)
            centroids[category] = self.F.normalize(centroid, dim=0)
        return centroids

    def word_embeddings(self, tokens: Sequence[str]):
        """Return one mean WordPiece embedding per original surface token."""
        torch = self.torch
        enc = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        word_ids = enc.word_ids(batch_index=0)
        enc_device = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            hidden = self.model(**enc_device).last_hidden_state[0]

        dim = hidden.shape[-1]
        word_vecs = []
        covered = []
        for wid in range(len(tokens)):
            idx = [j for j, w in enumerate(word_ids) if w == wid]
            if idx:
                vec = hidden[idx].mean(dim=0)
                word_vecs.append(self.F.normalize(vec, dim=0))
                covered.append(True)
            else:
                word_vecs.append(torch.zeros(dim, device=self.device))
                covered.append(False)
        return torch.stack(word_vecs, dim=0), covered

    def score_span(self, word_vecs, covered: Sequence[bool], start: int, end: int, category: str) -> Optional[float]:
        if category not in self.centroids:
            return None
        ids = [i for i in range(start, end) if i < len(covered) and covered[i]]
        if not ids:
            return None
        span = word_vecs[ids].mean(dim=0)
        if float(span.norm().item()) == 0:
            return None
        span = self.F.normalize(span, dim=0)
        return float(self.F.cosine_similarity(span.unsqueeze(0), self.centroids[category].unsqueeze(0)).item())

    def score_all_categories(
        self,
        word_vecs,
        covered: Sequence[bool],
        start: int,
        end: int,
    ) -> Dict[str, float]:
        """Score one span against every available category centroid."""
        scores: Dict[str, float] = {}
        for category in self.centroids:
            score = self.score_span(
                word_vecs, covered, start, end, category
            )
            if score is not None:
                scores[category] = score
        return scores

"""## Candidate selection / BIO generation"""

def add_context_scores(candidates: List[Candidate], encoder: ContextEncoder, word_vecs, covered):
    out = []
    for c in candidates:
        out.append(replace(c, context_score=encoder.score_span(word_vecs, covered, c.start, c.end, c.category)))
    return out

def add_category_margins(
    candidates: Sequence[Candidate],
    encoder: ContextEncoder,
    word_vecs,
    covered: Sequence[bool],
) -> List[Candidate]:
    """Attach target score, strongest competing score, and their margin.

    This uses only frozen contextual representations and lexicon-derived category
    centroids; it does not use gold annotations.
    """
    out: List[Candidate] = []
    for c in candidates:
        scores = encoder.score_all_categories(
            word_vecs, covered, c.start, c.end
        )
        target_score = scores.get(c.category, c.context_score)
        other_scores = [
            score for category, score in scores.items()
            if category != c.category
        ]
        strongest_other = max(other_scores) if other_scores else None
        margin = (
            target_score - strongest_other
            if target_score is not None and strongest_other is not None
            else None
        )
        out.append(
            replace(
                c,
                context_score=target_score,
                other_category_score=strongest_other,
                category_margin=margin,
            )
        )
    return out

def category_competition_filter(
    candidates: Sequence[Candidate],
    min_margin: float = 0.03,
) -> List[Candidate]:
    """Reject contextually ambiguous candidates using a category margin.

    Explicit multi-word exact lexicon entries are retained as strong lexical
    evidence. Candidates with no usable margin are also retained rather than
    being rejected for missing contextual coverage.
    """
    accepted: List[Candidate] = []
    for c in candidates:
        if c.match_type == "exact" and c.length > 1:
            accepted.append(c)
            continue
        if c.category_margin is None:
            accepted.append(c)
            continue
        if c.category_margin >= min_margin:
            accepted.append(c)
    return accepted

def category_specific_guard(
    candidate: Candidate,
    tokens: Sequence[str],
) -> bool:
    """Apply conservative category-specific sanity checks.

    V4 currently guards only the clearest phone false positives: bare numeric or
    k-suffixed price-like strings. More rules should be added only when justified
    independently of held-out test labels.
    """
    surface = " ".join(
        normalize_basic(t)
        for t in tokens[candidate.start:candidate.end]
    )

    if candidate.category == "phone":
        if re.fullmatch(r"\d+(?:\.\d+)?", surface):
            return False
        if re.fullmatch(r"\d+(?:\.\d+)?k", surface, flags=re.IGNORECASE):
            return False

    return True

def semantic_candidate_matches(
    tokens,
    normalizer,
    stopwords,
    sentiment,
    encoder,
    word_vecs,
    covered,
    occupied,
    score_threshold=0.60,
    margin_threshold=0.05,
):
    """
    Discover high-confidence single-token aspect candidates that
    were not recovered by lexical / fuzzy / regex matching.

    Uses only frozen contextual representations and category
    centroids. No gold labels are used.
    """

    proposals = []

    for i, token in enumerate(tokens):

        # Do not compete with an existing lexical candidate.
        if i in occupied:
            continue

        # mBERT may truncate very long reviews.
        if i >= len(covered) or not covered[i]:
            continue

        surface = normalize_basic(token)

        if not surface:
            continue

        # Avoid very short/noisy tokens.
        if len(surface) < 2:
            continue

        # Avoid pure numbers and price-like numeric strings.
        if re.fullmatch(
            r"\d+(?:\.\d+)?(?:k)?",
            surface,
            flags=re.IGNORECASE
        ):
            continue

        # Do not turn stopwords/sentiment words into aspects.
        if is_filtered_surface(
            token,
            stopwords,
            sentiment,
            normalizer
        ):
            continue

        # Compare this token against every category centroid.
        scores = encoder.score_all_categories(
            word_vecs,
            covered,
            i,
            i + 1
        )

        if len(scores) < 2:
            continue

        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        best_category, best_score = ranked[0]
        second_category, second_score = ranked[1]

        margin = best_score - second_score

        # Require both high absolute confidence and
        # clear separation from competing categories.
        if best_score < score_threshold:
            continue

        if margin < margin_threshold:
            continue

        candidate = Candidate(
            start=i,
            end=i + 1,
            category=best_category,
            canonical=surface,
            lexical_score=0.0,
            match_type="semantic",
            filtered=False,
            category_trigger=False,
            context_score=best_score,
            other_category_score=second_score,
            category_margin=margin,
        )

        # Reuse category-specific sanity checks.
        if category_specific_guard(
            candidate,
            tokens
        ):
            proposals.append(candidate)

    return proposals

def context_filter_candidates(
    candidates,
    exact_threshold,
    trigger_threshold,
    fuzzy_threshold
):
    accepted = []

    for c in candidates:

        score = c.context_score

        # Strong explicit multi-word lexicon entry
        if (
            c.match_type == "exact"
            and c.length > 1
        ):
            accepted.append(c)
            continue

        # Broad category-name trigger
        if c.category_trigger:

            if (
                score is not None
                and score >= trigger_threshold
            ):
                accepted.append(c)

            continue

        # Fuzzy lexical match
        if c.match_type == "fuzzy":

            if (
                score is not None
                and score >= fuzzy_threshold
            ):
                accepted.append(c)

            continue

        # Single-word exact / regex candidate
        if c.length == 1:

            if (
                score is not None
                and score >= exact_threshold
            ):
                accepted.append(c)

            continue

        accepted.append(c)

    return accepted

# def candidate_rank(c: Candidate) -> Tuple[float, float, float, int]:
#     # Longest exact lexical spans dominate. mBERT is used to resolve otherwise
#     # comparable/category-ambiguous candidates, not as a pseudo-supervised labeler.
#     context = c.context_score if c.context_score is not None else -1.0
#     trigger_bonus = 0.02 if c.category_trigger else 0.0
#     exact_bonus = 0.03 if c.match_type in {"exact", "regex"} else 0.0
#     return (float(c.length), c.lexical_score + trigger_bonus + exact_bonus, context, -c.start)

def candidate_rank(c: Candidate):

    context = (
        c.context_score
        if c.context_score is not None
        else -1.0
    )

    margin = (
        c.category_margin
        if c.category_margin is not None
        else -1.0
    )

    # Evidence strength
    if (
        c.match_type == "exact"
        and c.length > 1
    ):
        evidence = 5

    elif c.match_type == "exact":
        evidence = 4

    elif c.match_type == "regex":
        evidence = 3

    elif c.match_type == "fuzzy":
        evidence = 2

    elif c.match_type == "semantic":
        evidence = 1

    elif c.match_type == "contextual-extension":
        evidence = 0

    else:
        evidence = -1

    return (
        evidence,
        margin,
        context,
        c.lexical_score,
        -c.start
    )


def best_per_span(candidates: Sequence[Candidate]) -> List[Candidate]:
    grouped: Dict[Tuple[int, int], List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped[(c.start, c.end)].append(c)
    out = []
    for _, group in grouped.items():
        out.append(max(group, key=candidate_rank))
    return out

def overlaps(a: Candidate, b: Candidate) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def simple_candidate_rank(c: Candidate):
    """Ablation rank with no contextual/category evidence-aware ordering.

    It keeps only deterministic longest-first lexical overlap resolution so that
    the contribution of the evidence-aware rank can be measured.
    """
    return (float(c.length), float(c.lexical_score), -int(c.start))


def best_per_span_simple(candidates: Sequence[Candidate]) -> List[Candidate]:
    grouped: Dict[Tuple[int, int], List[Candidate]] = defaultdict(list)
    for c in candidates:
        grouped[(c.start, c.end)].append(c)
    return [max(group, key=simple_candidate_rank) for group in grouped.values()]


def select_nonoverlapping_simple(
    candidates: Sequence[Candidate]
) -> Tuple[List[Candidate], List[Candidate]]:
    """Longest-first lexical-only selector used only for the overlap ablation."""
    candidates = best_per_span_simple(candidates)
    primary = [c for c in candidates if not c.filtered]
    filtered = [c for c in candidates if c.filtered]
    selected: List[Candidate] = []
    for c in sorted(primary, key=simple_candidate_rank, reverse=True):
        if not any(overlaps(c, s) for s in selected):
            selected.append(c)
    return sorted(selected, key=lambda x: (x.start, x.end)), filtered

# def select_nonoverlapping(candidates: Sequence[Candidate]) -> Tuple[List[Candidate], List[Candidate]]:
#     """Greedy longest/highest-confidence selection, postponing fully filtered spans."""
#     candidates = best_per_span(candidates)
#     primary = [c for c in candidates if not c.filtered]
#     filtered = [c for c in candidates if c.filtered]

#     selected: List[Candidate] = []
#     for c in sorted(primary, key=candidate_rank, reverse=True):
#         if not any(overlaps(c, s) for s in selected):
#             selected.append(c)

#     # A sentiment-like lexical token may still be part of a compound when it is
#     # directly adjacent to a non-filtered aspect of the SAME category, e.g.
#     # "camera quality". It is never introduced as a standalone aspect.
#     for c in sorted(filtered, key=candidate_rank, reverse=True):
#         if c.length != 1 or any(overlaps(c, s) for s in selected):
#             continue
#         if any(s.category == c.category and (s.end == c.start or c.end == s.start) for s in selected):
#             selected.append(c)

#     return sorted(selected, key=lambda x: (x.start, x.end)), filtered

def select_nonoverlapping(
    candidates: Sequence[Candidate]
) -> Tuple[List[Candidate], List[Candidate]]:

    candidates = best_per_span(candidates)

    primary = [
        c for c in candidates
        if not c.filtered
    ]

    filtered = [
        c for c in candidates
        if c.filtered
    ]

    selected = []

    for c in sorted(
        primary,
        key=candidate_rank,
        reverse=True
    ):
        if not any(
            overlaps(c, s)
            for s in selected
        ):
            selected.append(c)

    return (
        sorted(
            selected,
            key=lambda x: (x.start, x.end)
        ),
        filtered
    )

def merge_adjacent_same_category(selected: Sequence[Candidate]) -> List[Candidate]:
    if not selected:
        return []
    items = sorted(selected, key=lambda c: (c.start, c.end))
    merged: List[Candidate] = []
    cur = items[0]
    for nxt in items[1:]:
        if cur.end == nxt.start and cur.category == nxt.category:
            canonical = (cur.canonical + " " + nxt.canonical).strip()
            ctx_vals = [v for v in (cur.context_score, nxt.context_score) if v is not None]
            cur = Candidate(
                start=cur.start,
                end=nxt.end,
                category=cur.category,
                canonical=canonical,
                lexical_score=min(cur.lexical_score, nxt.lexical_score),
                match_type="compound",
                filtered=False,
                category_trigger=cur.category_trigger or nxt.category_trigger,
                context_score=float(sum(ctx_vals) / len(ctx_vals)) if ctx_vals else None,
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    return merged

def contextual_neighbor_extension(
    selected: Sequence[Candidate],
    tokens: Sequence[str],
    stopwords: Set[str],
    sentiment: Set[str],
    normalizer: Normalizer,
    encoder: Optional[ContextEncoder],
    word_vecs,
    covered,
    threshold: float,
) -> List[Candidate]:
    """Conservative one-token contextual expansion for unlisted technical attributes.

    This is how the deterministic implementation can recover expressions such as
    "screen brightness" when `brightness` is absent from the supplied seed lexicon.
    The threshold is an implementation parameter because the manuscript does not
    state one for contextual cosine similarity.
    """
    if encoder is None or not selected:
        return list(selected)

    occupied = set()
    for s in selected:
        occupied.update(range(s.start, s.end))

    out = list(selected)
    proposals: List[Candidate] = []
    for s in selected:
        for idx, direction in ((s.start - 1, "left"), (s.end, "right")):
            if idx < 0 or idx >= len(tokens) or idx in occupied:
                continue
            # Do not contextually absorb function/sentiment words. Lexicon-backed
            # sentiment words were already handled in select_nonoverlapping().
            if is_filtered_surface(tokens[idx], stopwords, sentiment, normalizer):
                continue
            score = encoder.score_span(word_vecs, covered, idx, idx + 1, s.category)
            if score is not None and score >= threshold:
                proposals.append(Candidate(
                    start=idx, end=idx + 1, category=s.category,
                    canonical=normalize_basic(tokens[idx]), lexical_score=0.0,
                    match_type="contextual-extension", filtered=False,
                    context_score=score,
                ))

    for p in sorted(proposals, key=candidate_rank, reverse=True):
        if not any(overlaps(p, s) for s in out):
            out.append(p)
    return merge_adjacent_same_category(sorted(out, key=lambda c: (c.start, c.end)))

def llm_refine_hook(review_text: str, tokens: Sequence[str], spans: Sequence[Candidate]) -> List[Candidate]:
    """No-op by design.

    IDT_Revised.docx names an LLM-refinement stage but supplies no executable
    specification. Insert the recovered historical model/prompt here if available.
    Until then, returning the deterministic spans prevents an unverifiable step from
    being silently invented.
    """
    return list(spans)

def spans_to_bio(tokens: Sequence[str], spans: Sequence[Candidate]) -> List[Tuple[str, str]]:
    tags = ["O"] * len(tokens)
    for s in sorted(spans, key=lambda c: (c.start, c.end)):
        if s.start < 0 or s.end > len(tokens) or s.start >= s.end:
            continue
        tags[s.start] = f"B-{s.category}"
        for i in range(s.start + 1, s.end):
            tags[i] = f"I-{s.category}"
    return list(zip(tokens, tags))

def spans_to_aspect_dict(tokens: Sequence[str], spans: Sequence[Candidate]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for s in sorted(spans, key=lambda c: (c.start, c.end)):
        surface = " ".join(normalize_basic(t) for t in tokens[s.start:s.end]).strip()
        term = s.canonical if s.match_type not in {"compound", "contextual-extension"} else surface
        term = normalize_basic(term)
        if term and term not in out[s.category]:
            out[s.category].append(term)
    return dict(out)

def validate_iob2(pairs: Sequence[Tuple[str, str]]) -> None:
    prev = "O"
    for idx, (_, tag) in enumerate(pairs):
        if tag.startswith("I-"):
            cat = tag[2:]
            if prev not in {f"B-{cat}", f"I-{cat}"}:
                raise ValueError(f"Invalid IOB2 sequence at token {idx}: {tag} after {prev}")
        prev = tag

"""## End-to-end extraction"""

class ATEPipeline:
    def __init__(
        self,
        lexicon_path: str,
        suffix_paths: Sequence[str],
        stopword_path: str,
        sentiment_paths: Sequence[str],
        fuzzy_cutoff: Optional[float] = None,
        use_context: bool = True,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,

        exact_context_threshold: Optional[float] = None,
        trigger_context_threshold: Optional[float] = None,
        fuzzy_context_threshold: Optional[float] = None,
        category_margin_threshold: Optional[float] = None,
        enable_semantic_discovery: bool = False,
        semantic_score_threshold: Optional[float] = None,
        semantic_margin_threshold: Optional[float] = None,

        enable_neighbor_extension: bool = False,
        adjacent_context_threshold: Optional[float] = None,

        taxonomy: str = "evaluation16",
        require_wordnet: bool = True,
    ):
        required_values = {
            "fuzzy_cutoff": fuzzy_cutoff,
            "model_name": model_name,
            "max_length": max_length,
            "exact_context_threshold": exact_context_threshold,
            "trigger_context_threshold": trigger_context_threshold,
            "fuzzy_context_threshold": fuzzy_context_threshold,
            "category_margin_threshold": category_margin_threshold,
        }
        missing = [k for k, v in required_values.items() if v is None]
        if missing:
            raise ValueError(f"Missing pipeline configuration values: {missing}")
        if not (0.0 <= float(fuzzy_cutoff) <= 1.0):
            raise ValueError("fuzzy_cutoff must be in [0,1]")
        if enable_semantic_discovery and (semantic_score_threshold is None or semantic_margin_threshold is None):
            raise ValueError("Semantic discovery thresholds are required when semantic discovery is enabled")
        if enable_neighbor_extension and adjacent_context_threshold is None:
            raise ValueError("Neighbor-extension threshold is required when neighbor extension is enabled")
        self.by_category, self.entries = parse_aspect_lexicon(lexicon_path)
        if taxonomy not in {"evaluation16", "lexicon"}:
            raise ValueError("taxonomy must be 'evaluation16' or 'lexicon'")
        self.taxonomy = taxonomy
        self.ignored_lexicon_categories: List[str] = []
        if taxonomy == "evaluation16":
            all_cats = set(self.by_category)
            self.ignored_lexicon_categories = sorted(all_cats - EVALUATION_CATEGORIES)
            self.by_category = {
                c: es for c, es in self.by_category.items() if c in EVALUATION_CATEGORIES
            }
            self.entries = [e for e in self.entries if e.category in EVALUATION_CATEGORIES]
        self.matcher = LexiconMatcher(self.entries)
        suffixes = load_union(suffix_paths) | {normalize_basic(s) for s in CORE_MAL_SUFFIXES}
        self.normalizer = Normalizer(suffixes)
        self.basic_only_normalizer = BasicOnlyNormalizer()

        # Ablation switches. The full configured pipeline leaves every switch False.
        self.disable_morphology = False
        self.disable_stopword_sentiment = False
        self.disable_fuzzy = False
        self.disable_regex = False
        self.disable_category_guard = False
        self.disable_evidence_overlap = False

        if require_wordnet and not self.normalizer.wordnet_available:
            raise RuntimeError(
                "NLTK WordNet data is required by the manuscript methodology. "
                "Install it with: python -m nltk.downloader wordnet omw-1.4  "
                "(or use --allow-wordnet-fallback only for structural/debug runs)."
            )
        self.stopwords = {normalize_basic(w) for w in load_word_set(stopword_path)}
        self.sentiment = {normalize_basic(w) for w in load_union(sentiment_paths)}
        # self.fuzzy_cutoff = fuzzy_cutoff
        # self.adjacent_context_threshold = adjacent_context_threshold
        # self.fuzzy_context_threshold = fuzzy_context_threshold
        self.fuzzy_cutoff = fuzzy_cutoff

        self.exact_context_threshold = (
            exact_context_threshold
        )

        self.trigger_context_threshold = (
            trigger_context_threshold
        )

        self.fuzzy_context_threshold = (
            fuzzy_context_threshold
        )

        self.category_margin_threshold = (
            category_margin_threshold
        )

        self.enable_neighbor_extension = (
            enable_neighbor_extension
        )
        self.enable_semantic_discovery = (
            enable_semantic_discovery
        )

        self.semantic_score_threshold = (
            semantic_score_threshold
        )

        self.semantic_margin_threshold = (
            semantic_margin_threshold
        )

        self.adjacent_context_threshold = (
            adjacent_context_threshold
        )
        self.encoder: Optional[ContextEncoder] = None
        if use_context:
            self.encoder = ContextEncoder(
                model_name=model_name,
                by_category=self.by_category,
                device=device,
                max_length=max_length,
            )

    @property
    def categories(self) -> List[str]:
        return sorted(self.by_category)

    # def extract(self, review_text: str):
    #     clean = clean_review_text(review_text)
    #     tokens = surface_tokenize(clean)
    #     token_variants = [self.normalizer.variants(t) for t in tokens]

    #     raw = candidate_matches(
    #         matcher=self.matcher,
    #         tokens=tokens,
    #         token_variants=token_variants,
    #         stopwords=self.stopwords,
    #         sentiment=self.sentiment,
    #         normalizer=self.normalizer,
    #         fuzzy_cutoff=self.fuzzy_cutoff,
    #     )

    #     word_vecs = covered = None
    #     if self.encoder is not None and tokens:
    #         word_vecs, covered = self.encoder.word_embeddings(tokens)
    #         raw = add_context_scores(raw, self.encoder, word_vecs, covered)
    #         # Fuzzy matches are intentionally subjected to an additional semantic
    #         # sanity check. Exact/regex lexicon hits are never rejected by this
    #         # implementation parameter. The manuscript specifies the lexical
    #         # fuzzy cutoff but not a contextual cosine threshold.
    #         raw = [
    #             c for c in raw
    #             if c.match_type != "fuzzy"
    #             or c.context_score is None
    #             or c.context_score >= self.fuzzy_context_threshold
    #         ]

    #     selected, _ = select_nonoverlapping(raw)
    #     # selected = merge_adjacent_same_category(selected)

    #     # if self.encoder is not None and tokens:
    #     #     selected = contextual_neighbor_extension(
    #     #         selected=selected,
    #     #         tokens=tokens,
    #     #         stopwords=self.stopwords,
    #     #         sentiment=self.sentiment,
    #     #         normalizer=self.normalizer,
    #     #         encoder=self.encoder,
    #     #         word_vecs=word_vecs,
    #     #         covered=covered,
    #     #         threshold=self.adjacent_context_threshold,
    #     #     )

    #     if self.encoder is not None and tokens:

    #         word_vecs, covered = (
    #             self.encoder.word_embeddings(tokens)
    #         )

    #         raw = add_context_scores(
    #             raw,
    #             self.encoder,
    #             word_vecs,
    #             covered
    #         )

    #         raw = context_filter_candidates(
    #             raw,
    #             exact_threshold=self.exact_context_threshold,
    #             trigger_threshold=self.trigger_context_threshold,
    #             fuzzy_threshold=self.fuzzy_context_threshold
    #         )

    #     selected = llm_refine_hook(clean, tokens, selected)
    #     bio = spans_to_bio(tokens, selected)
    #     validate_iob2(bio)
    #     aspects = spans_to_aspect_dict(tokens, selected)
    #     return tokens, aspects, bio, selected

    def extract(self, review_text: str):

        # ---------------------------------------------------------
        # 1. Clean and tokenize review
        # ---------------------------------------------------------
        clean = clean_review_text(review_text)

        tokens = surface_tokenize(clean)

        active_normalizer = (
            self.basic_only_normalizer
            if self.disable_morphology
            else self.normalizer
        )

        token_variants = [
            active_normalizer.variants(t)
            for t in tokens
        ]


        # ---------------------------------------------------------
        # 2. Lexicon / fuzzy / regex candidate generation
        # ---------------------------------------------------------
        active_stopwords = set() if self.disable_stopword_sentiment else self.stopwords
        active_sentiment = set() if self.disable_stopword_sentiment else self.sentiment

        raw = candidate_matches(
            matcher=self.matcher,
            tokens=tokens,
            token_variants=token_variants,
            stopwords=active_stopwords,
            sentiment=active_sentiment,
            normalizer=active_normalizer,
            fuzzy_cutoff=self.fuzzy_cutoff,
        )

        if self.disable_fuzzy:
            raw = [c for c in raw if c.match_type != "fuzzy"]

        if self.disable_regex:
            raw = [c for c in raw if c.match_type != "regex"]


        # ---------------------------------------------------------
        # 3. Contextual scoring and filtering
        # ---------------------------------------------------------
        word_vecs = None
        covered = None

        if self.encoder is not None and tokens:

            # Get mBERT embeddings ONCE
            word_vecs, covered = (
                self.encoder.word_embeddings(tokens)
            )

            # Attach target-category context score to each candidate.
            raw = add_context_scores(
                raw,
                self.encoder,
                word_vecs,
                covered
            )

            # V4: compare the proposed category with all competing centroids.
            raw = add_category_margins(
                raw,
                self.encoder,
                word_vecs,
                covered
            )

            # Existing V3 contextual filtering.
            raw = context_filter_candidates(
                raw,
                exact_threshold=self.exact_context_threshold,
                trigger_threshold=self.trigger_context_threshold,
                fuzzy_threshold=self.fuzzy_context_threshold
            )

            # V4: remove category-ambiguous candidates.
            # raw = category_competition_filter(
            #     raw,
            #     min_margin=self.category_margin_threshold
            # )

            # Category-specific guard is applied below, outside the context block,
            # so the --no-context ablation removes only contextual encoding/filtering.

            # ---------------------------------------------------------
            # V4.2: high-confidence semantic candidate discovery
            # ---------------------------------------------------------
            if (
                self.enable_semantic_discovery
                and self.encoder is not None
                and tokens
            ):

                # Tokens already covered by surviving lexical candidates.
                occupied = set()

                for c in raw:
                    occupied.update(
                        range(c.start, c.end)
                    )

                semantic_candidates = semantic_candidate_matches(
                    tokens=tokens,
                    normalizer=self.normalizer,
                    stopwords=self.stopwords,
                    sentiment=self.sentiment,
                    encoder=self.encoder,
                    word_vecs=word_vecs,
                    covered=covered,
                    occupied=occupied,
                    score_threshold=self.semantic_score_threshold,
                    margin_threshold=self.semantic_margin_threshold,
                )

                raw.extend(semantic_candidates)


        # ---------------------------------------------------------
        # 3b. Category-specific sanity guard (independent component)
        # ---------------------------------------------------------
        if not self.disable_category_guard:
            raw = [c for c in raw if category_specific_guard(c, tokens)]

        # ---------------------------------------------------------
        # 4. Select non-overlapping candidates
        # IMPORTANT:
        # selection happens AFTER contextual filtering
        # ---------------------------------------------------------
        if self.disable_evidence_overlap:
            selected, _ = select_nonoverlapping_simple(raw)
        else:
            selected, _ = select_nonoverlapping(raw)


        # ---------------------------------------------------------
        # 5. DO NOT automatically merge adjacent candidates
        # ---------------------------------------------------------

        # DO NOT use:
        #
        # selected = merge_adjacent_same_category(selected)


        # ---------------------------------------------------------
        # 6. Optional contextual neighbour extension
        # Disabled by default in V3
        # ---------------------------------------------------------
        if (
            self.enable_neighbor_extension
            and self.encoder is not None
            and tokens
        ):

            selected = contextual_neighbor_extension(
                selected=selected,
                tokens=tokens,
                stopwords=self.stopwords,
                sentiment=self.sentiment,
                normalizer=self.normalizer,
                encoder=self.encoder,
                word_vecs=word_vecs,
                covered=covered,
                threshold=self.adjacent_context_threshold,
            )


        # ---------------------------------------------------------
        # 7. LLM refinement hook
        # Currently no-op
        # ---------------------------------------------------------
        selected = llm_refine_hook(
            clean,
            tokens,
            selected
        )


        # ---------------------------------------------------------
        # 8. Generate BIO tags
        # ---------------------------------------------------------
        bio = spans_to_bio(
            tokens,
            selected
        )

        validate_iob2(bio)


        # ---------------------------------------------------------
        # 9. Build aspect dictionary
        # ---------------------------------------------------------
        aspects = spans_to_aspect_dict(
            tokens,
            selected
        )


        return tokens, aspects, bio, selected

"""## CSV driver"""

def process_csv(
    pipeline: ATEPipeline,
    input_path: str,
    output_path: str,
    text_column: str = "Review_Text",
    limit: Optional[int] = None,
    diagnostics_path: Optional[str] = None,
) -> pd.DataFrame:
    src = pd.read_csv(input_path)
    if text_column not in src.columns:
        raise KeyError(f"Input CSV must contain column {text_column!r}. Found: {list(src.columns)}")
    if limit is not None:
        src = src.head(limit).copy()

    rows = []
    diagnostics = []
    total = len(src)
    for pos, (_, row) in enumerate(src.iterrows(), start=1):
        text = row[text_column]
        tokens, aspects, bio, spans = pipeline.extract(text)
        rows.append({
            "Review_Text": unicode_nfc(text),
            "Word_Tokens": repr(tokens),
            "Updated_Aspect_Terms": repr(aspects),
            "BIO_Tags": repr(bio),
        })
        if diagnostics_path:
            for s in spans:
                diagnostics.append({
                    "row": pos - 1,
                    "surface": " ".join(tokens[s.start:s.end]),
                    "start": s.start,
                    "end": s.end,
                    "category": s.category,
                    "canonical": s.canonical,
                    "match_type": s.match_type,
                    "lexical_score": s.lexical_score,
                    "context_score": s.context_score,
                    "other_category_score": s.other_category_score,
                    "category_margin": s.category_margin,
                })
        if pos == 1 or pos % 100 == 0 or pos == total:
            print(f"Processed {pos}/{total}")

    out = pd.DataFrame(rows, columns=["Review_Text", "Word_Tokens", "Updated_Aspect_Terms", "BIO_Tags"])
    out.to_csv(output_path, index=False, encoding="utf-8-sig")
    if diagnostics_path:
        pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    return out

def find_default(candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        if Path(name).exists():
            return name
    return None

# Default input candidates for auto-detection when --input is not supplied.
# ATE_bio_tags_input.csv is the primary input file containing Review_Text & Word_Tokens.
DEFAULT_INPUT_CANDIDATES = [
    "ATE_bio_tags_input.csv",
    "ATE_bio_tags.csv",
    "ATE_Bio_tags.csv",
    "ATE_bio_tags(1).csv",
]

def build_arg_parser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description=(
            "Generate category-specific ATE BIO tags "
            "from Manglish reviews."
        )
    )

    p.add_argument(
        "--input",
        required=True,
        help=(
            "CSV containing Review_Text. "
            "If omitted, auto-detect input."
        )
    )

    p.add_argument(
        "--output",
        required=True
    )

    p.add_argument(
        "--text-column",
        default="Review_Text"
    )

    p.add_argument(
        "--lexicon",
        required=True
    )

    p.add_argument(
        "--suffix-file",
        action="append",
        dest="suffix_files",
        required=True,
        help="Repeat for each Malayalam suffix file."
    )

    p.add_argument(
        "--stopwords",
        required=True
    )

    p.add_argument(
        "--sentiment-file",
        action="append",
        dest="sentiment_files",
        required=True,
        help=(
            "Repeat for positive/negative "
            "Malayalam/English lexicon files."
        )
    )

    # -----------------------------------------------------
    # Lexical matching
    # -----------------------------------------------------

    p.add_argument(
        "--fuzzy-cutoff",
        type=float,
        required=True
    )

    # -----------------------------------------------------
    # Contextual encoder
    # -----------------------------------------------------

    p.add_argument(
        "--model",
        required=True
    )

    p.add_argument(
        "--device",
        default=None,
        help="e.g. cuda, cuda:0, cpu"
    )

    p.add_argument(
        "--max-length",
        type=int,
        required=True
    )

    # -----------------------------------------------------
    # V3 precision controls
    # -----------------------------------------------------

    p.add_argument(
        "--exact-context-threshold",
        type=float,
        required=True,
        help=(
            "Minimum contextual score for "
            "single-word exact candidates."
        )
    )

    p.add_argument(
        "--trigger-context-threshold",
        type=float,
        required=True,
        help=(
            "Minimum contextual score for "
            "category-header triggers."
        )
    )

    p.add_argument(
        "--fuzzy-context-threshold",
        type=float,
        required=True,
        help=(
            "Minimum contextual score for "
            "fuzzy candidates."
        )
    )

    p.add_argument(
        "--category-margin-threshold",
        type=float,
        required=True,
        help=(
            "Minimum target-category score minus strongest competing "
            "category score for contextually ambiguous candidates."
        )
    )

    p.add_argument(
    "--enable-semantic-discovery",
    action="store_true",
    help=(
        "Enable V4.2 semantic discovery for tokens "
        "not covered by lexical candidates."
        )
    )

    p.add_argument(
        "--semantic-score-threshold",
        type=float,
        default=None,
        help=(
            "Minimum category-centroid cosine score "
            "for a semantic candidate."
        )
    )

    p.add_argument(
        "--semantic-margin-threshold",
        type=float,
        default=None,
        help=(
            "Minimum best-vs-second-category margin "
            "for a semantic candidate."
        )
    )


    p.add_argument(
        "--enable-neighbor-extension",
        action="store_true",
        help=(
            "Enable contextual one-token neighbour "
            "extension. Disabled by default in V3."
        )
    )

    p.add_argument(
        "--adjacent-context-threshold",
        type=float,
        default=None,
        help=(
            "Minimum contextual score for "
            "neighbour extension."
        )
    )

    # -----------------------------------------------------
    # component-ablation switches
    # Full pipeline = leave every switch below OFF.
    # -----------------------------------------------------
    p.add_argument("--disable-morphology", action="store_true",
                   help="Ablation: use only basic NFC/lowercase forms; disable suffix stripping and English lemmatization.")
    p.add_argument("--disable-stopword-sentiment", action="store_true",
                   help="Ablation: disable stopword/sentiment candidate suppression.")
    p.add_argument("--disable-fuzzy", action="store_true",
                   help="Ablation: remove all fuzzy lexical candidates.")
    p.add_argument("--disable-regex", action="store_true",
                   help="Ablation: remove all regex-trigger candidates.")
    p.add_argument("--disable-category-guard", action="store_true",
                   help="Ablation: disable category-specific sanity checks.")
    p.add_argument("--disable-evidence-overlap", action="store_true",
                   help="Ablation: replace evidence-aware overlap resolution with longest-first lexical-only selection.")

    # -----------------------------------------------------
    # Other options
    # -----------------------------------------------------

    p.add_argument(
        "--taxonomy",
        choices=[
            "evaluation16",
            "lexicon"
        ],
        default="evaluation16",
        help=(
            "evaluation16 = manuscript-compatible "
            "label inventory; lexicon = keep all "
            "lexicon categories."
        )
    )

    p.add_argument(
        "--allow-wordnet-fallback",
        action="store_true",
        help=(
            "Debug only: allow fallback English "
            "normalization if WordNet is unavailable."
        )
    )

    p.add_argument(
        "--no-context",
        action="store_true",
        help="Debug mode: skip mBERT contextual encoding."
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None
    )

    p.add_argument(
        "--diagnostics",
        default=None,
        help="Optional candidate diagnostics CSV."
    )

    return p

def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    # When input and output resolve to the same file, read the input first then
    # overwrite it with the regenerated tags.
    input_resolved = Path(args.input).resolve()
    output_resolved = Path(args.output).resolve()
    if input_resolved == output_resolved:
        import tempfile, shutil
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="ate_input_")
        os.close(tmp_fd)
        shutil.copy2(str(input_resolved), tmp_path)
        args.input = tmp_path
        _cleanup_tmp = tmp_path  # will be deleted after processing
    else:
        _cleanup_tmp = None

    suffix_files = list(args.suffix_files)
    sentiment_files = list(args.sentiment_files)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    print("Building weakly supervised ATE pipeline...")
    print("Building ATE pipeline...")
    pipeline = ATEPipeline(
        lexicon_path=args.lexicon,
        suffix_paths=suffix_files,
        stopword_path=args.stopwords,
        sentiment_paths=sentiment_files,
        fuzzy_cutoff=args.fuzzy_cutoff,
        use_context=not args.no_context,
        model_name=args.model,
        device=args.device,
        max_length=args.max_length,
        exact_context_threshold=args.exact_context_threshold,
        trigger_context_threshold=args.trigger_context_threshold,
        fuzzy_context_threshold=args.fuzzy_context_threshold,
        category_margin_threshold=args.category_margin_threshold,
        enable_semantic_discovery=args.enable_semantic_discovery,
        semantic_score_threshold=args.semantic_score_threshold,
        semantic_margin_threshold=args.semantic_margin_threshold,
        enable_neighbor_extension=args.enable_neighbor_extension,
        adjacent_context_threshold=args.adjacent_context_threshold,
        taxonomy=args.taxonomy,
        require_wordnet=not args.allow_wordnet_fallback,
    )
    pipeline.disable_morphology = args.disable_morphology
    pipeline.disable_stopword_sentiment = args.disable_stopword_sentiment
    pipeline.disable_fuzzy = args.disable_fuzzy
    pipeline.disable_regex = args.disable_regex
    pipeline.disable_category_guard = args.disable_category_guard
    pipeline.disable_evidence_overlap = args.disable_evidence_overlap

    active_ablation = []
    if args.no_context:
        active_ablation.append("NO_CONTEXT")
    if args.disable_morphology:
        active_ablation.append("NO_MORPHOLOGY")
    if args.disable_stopword_sentiment:
        active_ablation.append("NO_STOPWORD_SENTIMENT")
    if args.disable_fuzzy:
        active_ablation.append("NO_FUZZY")
    if args.disable_regex:
        active_ablation.append("NO_REGEX")
    if args.disable_category_guard:
        active_ablation.append("NO_CATEGORY_GUARD")
    if args.disable_evidence_overlap:
        active_ablation.append("NO_EVIDENCE_OVERLAP")

    print("### ATE PIPELINE CONFIGURATION ###")
    print("Ablation:", ", ".join(active_ablation) if active_ablation else "FULL_PIPELINE")
    print("Thresholds:", args.exact_context_threshold, args.trigger_context_threshold, args.fuzzy_context_threshold)
    print("Categories:", ", ".join(pipeline.categories))
    if pipeline.ignored_lexicon_categories:
        print("Ignored lexicon-only categories under evaluation16 taxonomy:",
              ", ".join(pipeline.ignored_lexicon_categories))
    print("WordNet lemmatizer:", "available" if pipeline.normalizer.wordnet_available else "fallback rules")
    print("Contextual encoder:", "disabled" if args.no_context else args.model)

    process_csv(
        pipeline=pipeline,
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        limit=args.limit,
        diagnostics_path=args.diagnostics,
    )
    print(f"Saved: {args.output}")
    if args.diagnostics:
        print(f"Saved diagnostics: {args.diagnostics}")

    # Clean up temporary input copy if we used one.
    if _cleanup_tmp and Path(_cleanup_tmp).exists():
        os.remove(_cleanup_tmp)
        print("(Cleaned up temporary input copy.)")

if __name__ == "__main__":
    main()

