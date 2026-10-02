# GitHub upload notes

## Use these files
- `code/ate_pipeline_core.py`
- `code/generate_ate_bio.py`
- `config/protocol.json`
- `config/pipeline_config.json`
- `config/transformer_config.json`
- `config/char_bilstm_config.json`
- `notebooks/final_run.ipynb`

## Do NOT upload `generate_ate_bio_tags_v4.py` as executable source

The supplied `generate_ate_bio_tags_v4.py` is an exported exploratory Colab
notebook rather than the clean generator. It contains machine-specific Drive
paths, old intermediate/result filenames, validation sweeps, V3/V4 comparison
cells and notebook shell syntax. It also does not compile as a standalone
Python module in its current form.

The clean executable pair is `generate_ate_bio.py` + `ate_pipeline_core.py`.

## Required data names in the repository
- `data/evaluation_input.csv` -- exactly the 1,000 reviews used by the pipeline
- `data/evaluation_gold.csv` -- the corresponding 1,000-review gold BIO file
- `data/dual_regime_protocol_manifest.csv`
- `data/fixed_test_manifest_seed42.csv`

## Required supporting scripts
Add the final versions already used in the study:
- `code/evaluate_ate.py`
- `code/compare_methods.py`
- `code/train_transformer_baselines.py`
- `code/train_char_bilstm_crf.py`
- `code/ablation_study.py`

## Required resources
Place the lexicon and linguistic resources under `resources/`.

Before pushing, run `notebooks/final_run.ipynb` from a clean Colab runtime.


## Environment files included
- `requirements.txt`
- `.gitignore`

The repository still needs the project-specific `data/` and `resources/` files
listed by the notebook. Run the notebook from a fresh Colab runtime before the
first public push.
