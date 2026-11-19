# Reproducibility record

## Source and output layout

| Directory | Content |
|---|---|
| `protocol/` | Fixed seed-42 manifests, 200-row aligned test input, original lexicon and pipeline configuration |
| `results/baselines/` | Selected fixed-test word-level predictions and strict-span metrics for four architectures in both supervision regimes |
| `results/pipeline/` | Original and pruned-lexicon fixed-test predictions, metrics, per-category scores and run metadata |
| `results/ablation/` | Seven original fixed-test component configurations and prediction files, plus their measured scores |
| `results/lexicon/` | 18-category lexical inventory, distinct-literal counts, operational exact and fuzzy exposure, candidate-stage coverage, pruning sensitivity |
| `results/corpus/` | Same-tokenizer and script-based corpus comparison tabulations |
| `results/statistics/` | Paired bootstrap, paired randomization, paired ablation intervals and five-seed baseline summary |
| `scripts/` | Fixed protocol/inference auditing, lexical exposure, selected baseline-run and statistical-validation programs |

## Exact definitions of lexical exposure

The 560 lexical-entry lines span 18 category headings. After excluding regex and category-heading-trigger entries and deduplicating normalized literal forms, the operational inventory contains 517 forms. Under **operational exact matching**, across the 1,000 annotated evaluation reviews and 3,883 text-disjoint source reviews, the categories are: 239 in both portions, 102 in the source only, 17 in evaluation only and 159 in neither. Fuzzy matching produces a different distribution; results from distinct matching definitions should not be interchanged. The post-hoc pruned inventory contains 500 forms.

On the 1,000 annotated reviews, candidate generation covered 755 of 791 gold category-strict spans (95.45%) and 785 of 791 gold boundaries (99.24%, category disregarded). The fixed-200 original model achieved P=0.3709677, R=0.8098592, Micro-F1=0.5088496. After post-hoc removal of the 17 evaluation-exclusive forms, P=0.3733766, R=0.8098592, Micro-F1=0.5111111. Original/pruned true positives remained 115; false positives changed from 195 to 193. Lexical pruning is a sensitivity diagnostic and does not reconstruct the historical lexicon.

## Inference, training and scoring

The existing repository `code/` and `resources/` directories provide the operational pipeline. `scripts/01_prepare_fixed_protocol.py` is a manifest-alignment check and prepares the original 200-row fixed-test input. `scripts/11_regenerate_component_ablation.py` re-executes the seven archived ablation configurations against the existing repository pipeline and a fresh output directory; the archived outputs remain untouched. The final baseline results are the selected frozen-split outputs under `results/baselines/`; the corresponding archived training program is `scripts/run_dual_supervision_baselines_EXACT_PREVIOUS_ORDER.py`. The pipeline is configured at exact/regex=0.35, category-heading trigger=0.60, fuzzy=0.55, with fuzzy string similarity >=0.75; multiword exact lexical candidates bypass the contextual retention cutoff, and overlapping selected spans are rejected. The category competition margin contributes to ranking rather than acting as an active rejection threshold.

The historical fixed-test baseline training configuration used AdamW, learning rate 5e-5, batch size 8, maximum 50 epochs and patience 5 for the transformers. The selected BiLSTM+CRF run used Adam, learning rate 1e-3, batch size 16 and five epochs. Different experiment archives record Transformers 5.16.1 and 5.17.0; the accompanying selected-run metadata distinguish them.

## Verified aggregates

- Original low-resource XLM-R: TP=67, FP=76, FN=75; Micro-F1=0.4701754.
- Original pipeline: TP=115, FP=195, FN=27; Micro-F1=0.5088496.
- Paired Micro-F1 delta +0.038674; two-sided review-level randomization p=0.3912.
- Paired recall delta +0.338028; exact paired sign-flip p<0.001.
- Paired bootstrap 95% F1 difference interval [-0.0472, 0.1259], and recall difference interval [0.2319, 0.4435].
- Five-seed fixed-split low-resource baseline Micro-F1 mean +/- SD: XLM-R 0.4357 +/- 0.1010, mBERT 0.2132 +/- 0.2263 and IndicBERT 0.1291 +/- 0.1070. The best XLM-R seed was 0.5404. The `seed_stability_summary.csv` is the supplied aggregate record; individual per-seed output files are not included in this update.
- Among component-removal tests, removing contextual filtering lowered Micro-F1 by approximately 0.0790, and removing the stopword/sentiment filter lowered it by 0.0275. Removing fuzzy matching and evidence-aware overlap resolution slightly increased F1 on this fixed evaluation.

These are fixed-test, post-hoc analyses, not a cross-domain evaluation or an independent lexicon-construction study. The `FILE_SHA256.csv` file establishes content integrity of this supplied archive; Git hosting, repository commit identity and any later immutable external archive are separate publication metadata.

Run `python reproducibility/verify_evidence.py` from the repository root to check the frozen evidence files.
