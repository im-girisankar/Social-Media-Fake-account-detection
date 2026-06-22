# fakedetect — CLAUDE.md

Developer notes for AI assistants working on this project.

## Key invariants — do not break

1. **TensorFlow is optional and lazily imported.**
   `pytest` must pass with zero network access and without TensorFlow installed.
   The tested core (`data.py`, `features.py`, `models.py` baselines, `evaluate.py`)
   uses only `numpy`, `pandas`, and `scikit-learn`.
   `build_keras_mlp` and `get_keras_callbacks` in `models.py` must be behind a
   lazy `import tensorflow` that raises a helpful `ImportError` when TF is absent.

2. **No data leakage.**
   `Preprocessor.fit_transform(X_train, X_test)` fits `StandardScaler` on
   `X_train` only, then calls `.transform()` (not `.fit_transform()`) on
   `X_test`.  The original notebook had `scaler.fit_transform(X_test)` — that
   bug is fixed and must remain fixed.  `tests/test_features.py` contains a
   regression test for this.

3. **Deterministic seeds everywhere.**
   `generate_synthetic(seed=42)`, `build_logreg(seed=42)`,
   `build_random_forest(seed=42)`, `build_keras_mlp(seed=42)` must all produce
   identical results across repeated runs.

4. **This is a feed-forward NN, not an LSTM.**
   The original notebook was sometimes mislabelled "LSTM". The architecture
   (`Dense → Dense → Dropout → Dense → Dropout → Dense`) has no recurrent
   component. Never call it an LSTM.

## Running checks

```bash
# Lint
python -m ruff check src tests

# Tests (no TF required)
python -m pytest

# End-to-end on synthetic data (no CSVs, no TF)
python -m fakedetect.cli train --synthetic
```

## Getting the real dataset

Download from Kaggle: "Instagram fake spammer genuine accounts"
https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts

Place `insta_train.csv` and `insta_test.csv` in the `data/` directory, then run:

```bash
python -m fakedetect.cli train
```

## Optional Keras MLP

```bash
pip install 'fakedetect[nn]'
python -m fakedetect.cli train --synthetic --nn
```
