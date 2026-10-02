"""
Inter-annotator agreement for the Manglish ATE BIO annotation.

Three annotators (A1, A2, A3) independently labelled the same 1,000 reviews at
token level under IOB2. This script computes, entirely from the annotation
files:

  * raw percent agreement and unanimity
  * pairwise Cohen's kappa  (the statistic the manuscript currently reports)
  * Fleiss' kappa           (the correct multi-rater statistic for 3 raters)
  * Krippendorff's alpha    (nominal)
  * the same statistics collapsed to a binary aspect / non-aspect decision
  * span-level exact-match agreement, which is what a span-level IOB2
    evaluation protocol actually rests on
  * cluster bootstrap 95% CIs, resampling whole reviews rather than tokens,
    since tokens within a review are not independent

Every statistic here is additive over reviews, so each review's sufficient
statistics are precomputed once and the bootstrap becomes a weighted sum over
them. That is not an approximation of the naive resampling loop -- it is
algebraically the same number, just cheap enough to run 10,000 times.
"""

import itertools
import numpy as np
import pandas as pd

MERGED = "/home/claude/merged_renamed.csv"
ANNOTATORS = ["A1", "A2", "A3"]
PAIRS = list(itertools.combinations(ANNOTATORS, 2))
N_BOOT = 10_000

# The bootstrap needs a generator. Seeding it from the OS entropy pool and
# printing the seed keeps the run both un-cherry-picked and exactly
# reproducible: pass the printed value back in to reproduce it bit for bit.
SEED = int(np.random.SeedSequence().entropy % (2 ** 32))


# ---------------------------------------------------------------- statistics


def cohen_from_cm(cm):
    """Cohen's kappa from a confusion matrix."""
    n = cm.sum()
    po = np.trace(cm) / n
    pe = (cm.sum(-1) * cm.sum(-2)).sum() / n ** 2
    return (po - pe) / (1 - pe)


def fleiss_from_stats(sq, cat_totals, n_items, m=3):
    """Fleiss' kappa from per-item sums of squared rating counts."""
    p_bar = (sq - n_items * m) / (n_items * m * (m - 1))
    p_j = cat_totals / (n_items * m)
    p_e = (p_j ** 2).sum()
    return (p_bar - p_e) / (1 - p_e)


def alpha_from_coinc(co):
    """Nominal Krippendorff's alpha from a coincidence matrix."""
    n_c = co.sum(-1)
    n = n_c.sum()
    do = n - np.trace(co)
    de = (n ** 2 - (n_c ** 2).sum()) / (n - 1)
    return 1 - do / de


def f1_from_counts(tp, fp, fn):
    p = tp / np.maximum(tp + fp, 1e-12)
    r = tp / np.maximum(tp + fn, 1e-12)
    return np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)


# ------------------------------------------------------------------- helpers


def spans(tags):
    """(start, end, category) spans from an IOB2 sequence."""
    out, start, cat = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t.startswith("B-"):
            if start is not None:
                out.append((start, i, cat))
            start, cat = i, t[2:]
        elif t.startswith("I-") and start is not None and t[2:] == cat:
            continue
        else:
            if start is not None:
                out.append((start, i, cat))
            start, cat = None, None
    return set(out)


def per_review_stats(groups, codes, k):
    """Sufficient statistics for every review, for one label encoding."""
    n_rev = len(groups)
    cms = np.zeros((len(PAIRS), n_rev, k, k))
    coinc = np.zeros((n_rev, k, k))
    sq = np.zeros(n_rev)
    cat_totals = np.zeros((n_rev, k))
    n_tok = np.zeros(n_rev)
    unanimous = np.zeros(n_rev)
    pair_idx = [(ANNOTATORS.index(a), ANNOTATORS.index(b)) for a, b in PAIRS]

    for r, idx in enumerate(groups):
        lab = codes[idx]                       # (tokens, 3)
        n_tok[r] = len(idx)
        for p, (i, j) in enumerate(pair_idx):
            np.add.at(cms[p, r], (lab[:, i], lab[:, j]), 1)
        counts = np.zeros((len(idx), k))
        np.add.at(counts, (np.arange(len(idx))[:, None], lab), 1)
        sq[r] = (counts ** 2).sum()
        cat_totals[r] = counts.sum(0)
        coinc[r] = (counts.T @ counts - np.diag(counts.sum(0))) / 2.0
        unanimous[r] = ((lab[:, 0] == lab[:, 1]) & (lab[:, 1] == lab[:, 2])).sum()

    return dict(cms=cms, coinc=coinc, sq=sq, cat_totals=cat_totals,
                n_tok=n_tok, unanimous=unanimous)


def evaluate(st, w):
    """All token-level statistics under review weights w (all ones = point est)."""
    cms = np.einsum("r,prij->pij", w, st["cms"])
    out = {}
    for p, (a, b) in enumerate(PAIRS):
        out[f"cohen_{a}{b}"] = cohen_from_cm(cms[p])
        out[f"agree_{a}{b}"] = np.trace(cms[p]) / cms[p].sum()
    out["cohen_mean"] = float(np.mean([out[f"cohen_{a}{b}"] for a, b in PAIRS]))
    n_tok = w @ st["n_tok"]
    out["fleiss"] = fleiss_from_stats(w @ st["sq"], w @ st["cat_totals"], n_tok)
    out["alpha"] = alpha_from_coinc(np.einsum("r,rij->ij", w, st["coinc"]))
    out["unanimous"] = (w @ st["unanimous"]) / n_tok
    return out


def ci(draws, key):
    v = np.array([d[key] for d in draws])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


# ---------------------------------------------------------------------- main


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(MERGED)
    df = df.sort_values(["Review_ID", "Token_Index"]).reset_index(drop=True)

    groups = [np.asarray(v) for v in
              df.groupby("Review_ID", sort=False).indices.values()]
    n_rev = len(groups)

    print("=" * 78)
    print("INTER-ANNOTATOR AGREEMENT - Manglish ATE BIO annotation")
    print("=" * 78)
    print(f"Annotators : {', '.join(ANNOTATORS)} (all labelled the same material)")
    print(f"Reviews    : {n_rev:,}")
    print(f"Tokens     : {len(df):,}")
    print(f"Bootstrap  : {N_BOOT:,} cluster resamples over reviews, seed = {SEED}")

    full_labels = sorted(set(df[ANNOTATORS].values.ravel()))
    full_map = {c: i for i, c in enumerate(full_labels)}
    full_codes = df[ANNOTATORS].map(lambda c: full_map[c]).values
    bin_codes = (df[ANNOTATORS].values != "O").astype(int)
    print(f"Label set  : {len(full_labels)} IOB2 tags; "
          f"{int(bin_codes.any(1).sum()):,} tokens called an aspect by >=1 annotator")

    weights = rng.multinomial(
        n_rev, np.full(n_rev, 1 / n_rev), size=N_BOOT).astype(float)

    for header, codes, k in [("FULL IOB2 LABEL SET", full_codes, len(full_labels)),
                             ("BINARY: aspect token vs. O", bin_codes, 2)]:
        st = per_review_stats(groups, codes, k)
        pt = evaluate(st, np.ones(n_rev))
        draws = [evaluate(st, w) for w in weights]

        print("\n" + "-" * 78)
        print(header)
        print("-" * 78)
        print(f"{'Pair':<14}{'Raw agreement':>16}{'Cohen kappa':>14}{'95% CI':>24}")
        for a, b in PAIRS:
            lo, hi = ci(draws, f"cohen_{a}{b}")
            print(f"{a + '-' + b:<14}{pt[f'agree_{a}{b}']:>16.4f}"
                  f"{pt[f'cohen_{a}{b}']:>14.4f}{f'[{lo:.4f}, {hi:.4f}]':>24}")
        for key, label in [("cohen_mean", "Mean pairwise Cohen"),
                           ("fleiss", "Fleiss kappa (3 raters)"),
                           ("alpha", "Krippendorff alpha")]:
            lo, hi = ci(draws, key)
            print(f"{label:<28}{pt[key]:>16.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"{'Unanimous tokens':<28}{pt['unanimous']:>16.4f}")

    # span level ------------------------------------------------------------
    print("\n" + "-" * 78)
    print("SPAN-LEVEL EXACT-MATCH AGREEMENT (no chance correction is defined)")
    print("-" * 78)
    tp = np.zeros((len(PAIRS), n_rev))
    fp = np.zeros((len(PAIRS), n_rev))
    fn = np.zeros((len(PAIRS), n_rev))
    tags = df[ANNOTATORS].values
    for r, idx in enumerate(groups):
        sets = [spans(tags[idx, i]) for i in range(3)]
        for p, (a, b) in enumerate(PAIRS):
            sa, sb = sets[ANNOTATORS.index(a)], sets[ANNOTATORS.index(b)]
            tp[p, r] = len(sa & sb)
            fn[p, r] = len(sa - sb)
            fp[p, r] = len(sb - sa)

    print(f"{'Pair':<14}{'F1':>10}{'95% CI':>24}{'Agreed':>9}{'Only A':>9}{'Only B':>9}")
    pt_f1 = []
    for p, (a, b) in enumerate(PAIRS):
        f = float(f1_from_counts(tp[p].sum(), fp[p].sum(), fn[p].sum()))
        boot = f1_from_counts(weights @ tp[p], weights @ fp[p], weights @ fn[p])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pt_f1.append(f)
        print(f"{a + '-' + b:<14}{f:>10.4f}{f'[{lo:.4f}, {hi:.4f}]':>24}"
              f"{int(tp[p].sum()):>9}{int(fn[p].sum()):>9}{int(fp[p].sum()):>9}")
    print(f"{'Mean':<14}{np.mean(pt_f1):>10.4f}")

    # disagreement profile --------------------------------------------------
    print("\n" + "-" * 78)
    print("DISAGREEMENT PROFILE")
    print("-" * 78)
    dis = df[~((df.A1 == df.A2) & (df.A2 == df.A3))]
    maj = dis[(dis.A1 == dis.A2) | (dis.A2 == dis.A3) | (dis.A1 == dis.A3)]
    print(f"Tokens with any disagreement      : {len(dis):,} ({len(dis)/len(df):.2%})")
    print(f"  resolvable by 2-1 majority      : {len(maj):,} ({len(maj)/max(len(dis),1):.1%})")
    print(f"  three-way split, no majority    : {len(dis)-len(maj):,}")
    print(f"Reviews with >=1 disagreement     : {dis.Review_ID.nunique():,} of {n_rev:,}")
    for a in ANNOTATORS:
        flag = df[f"{a}_amb"].astype(str).str.strip().str.lower().eq("yes").sum()
        print(f"Tokens flagged 'Ambiguous' by {a} : {flag:,}")

    print("\nMost frequent disagreement patterns (A1 / A2 / A3):")
    for (t1, t2, t3), n in dis.groupby(ANNOTATORS).size().sort_values(
            ascending=False).head(12).items():
        print(f"  {n:>5}   {t1:<16}{t2:<16}{t3}")

    dis.to_csv("/home/claude/disagreements_renamed.csv", index=False)
    print(f"\nSaved {len(dis):,} disagreement rows -> disagreements.csv")


if __name__ == "__main__":
    main()
