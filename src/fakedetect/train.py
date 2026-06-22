"""End-to-end training pipeline.

Loads data (real CSVs or synthetic fallback), preprocesses with no data
leakage, trains baselines (+ optional Keras MLP), evaluates, and prints a
comparison report.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from fakedetect.data import FEATURE_COLUMNS, generate_synthetic, load_csv
from fakedetect.evaluate import evaluate, format_report
from fakedetect.features import Preprocessor, split_xy
from fakedetect.models import build_logreg, build_random_forest


def run_pipeline(
    data_dir: str | Path | None = None,
    use_synthetic: bool = False,
    train_nn: bool = False,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Run the full training and evaluation pipeline.

    Parameters
    ----------
    data_dir:
        Directory containing ``insta_train.csv`` and ``insta_test.csv``.
        When ``None`` **and** ``use_synthetic`` is ``False``, defaults to
        ``data/`` relative to the project root (auto-detects).
    use_synthetic:
        Force synthetic data even if real CSVs exist.
    train_nn:
        Also train the optional Keras MLP (requires TensorFlow).
    seed:
        Global random seed for sklearn models.

    Returns
    -------
    dict[str, dict]
        Mapping of model name → evaluation metrics dict.
    """
    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if use_synthetic:
        print("[data] Using synthetic dataset (no real CSVs needed).")
        train_df, test_df = generate_synthetic(seed=seed)
    else:
        data_dir = Path(data_dir) if data_dir else _find_data_dir()
        train_csv = data_dir / "insta_train.csv"
        test_csv = data_dir / "insta_test.csv"

        if train_csv.exists() and test_csv.exists():
            print(f"[data] Loading real CSVs from {data_dir}")
            train_df, test_df = load_csv(train_csv, test_csv)
        else:
            print(
                f"[data] Real CSVs not found in {data_dir}. "
                "Falling back to synthetic dataset."
            )
            train_df, test_df = generate_synthetic(seed=seed)

    print(f"[data] Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    # ------------------------------------------------------------------ #
    # 2. Split X/y and preprocess (no leakage)
    # ------------------------------------------------------------------ #
    X_train, y_train = split_xy(train_df)
    X_test, y_test = split_xy(test_df)

    prep = Preprocessor()
    X_train_s, X_test_s = prep.fit_transform(X_train, X_test)
    print(
        f"[preprocess] Scaler fitted on train only "
        f"(mean[0]={prep.train_mean_[0]:.4f}, scale[0]={prep.train_scale_[0]:.4f})."
    )

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    results: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # 3. Logistic Regression
    # ------------------------------------------------------------------ #
    print("\n[model] Training Logistic Regression ...")
    lr = build_logreg(seed=seed)
    lr.fit(X_train_s, y_train_np)
    lr_pred = lr.predict(X_test_s)
    lr_proba = lr.predict_proba(X_test_s)[:, 1]
    results["Logistic Regression"] = evaluate(y_test_np, lr_pred, lr_proba)

    # ------------------------------------------------------------------ #
    # 4. Random Forest
    # ------------------------------------------------------------------ #
    print("[model] Training Random Forest ...")
    rf = build_random_forest(seed=seed)
    rf.fit(X_train_s, y_train_np)
    rf_pred = rf.predict(X_test_s)
    rf_proba = rf.predict_proba(X_test_s)[:, 1]
    results["Random Forest"] = evaluate(y_test_np, rf_pred, rf_proba)

    # ------------------------------------------------------------------ #
    # 5. Optional Keras MLP
    # ------------------------------------------------------------------ #
    if train_nn:
        results.update(_run_keras(X_train_s, y_train_np, X_test_s, y_test_np, seed))

    # ------------------------------------------------------------------ #
    # 6. Print comparison report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    for name, metrics in results.items():
        print(format_report(name, metrics))
        print()

    # Feature importance from Random Forest
    rf_model = rf
    importances = rf_model.feature_importances_
    print("Random Forest feature importances:")
    for feat, imp in sorted(zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]):
        print(f"  {feat:<28} {imp:.4f}")

    return results


def _run_keras(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Train the optional Keras MLP and return its metrics."""
    from fakedetect.models import build_keras_mlp, get_keras_callbacks  # noqa: PLC0415

    try:
        import tensorflow as tf  # noqa: PLC0415

        tf.keras.utils.set_random_seed(seed)

        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=2)

        model = build_keras_mlp(input_dim=X_train.shape[1], seed=seed)
        callbacks = get_keras_callbacks(patience=15)

        print("\n[model] Training Keras MLP (EarlyStopping active) ...")
        model.fit(
            X_train,
            y_train_cat,
            epochs=200,
            batch_size=32,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=0,
        )

        y_proba_cat = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_proba_cat, axis=1)
        y_proba = y_proba_cat[:, 1]

        # Compare against y_test (integer labels)
        metrics = evaluate(y_test, y_pred, y_proba)
        return {"Keras MLP": metrics}

    except ImportError:
        print(
            "[model] TensorFlow not installed — skipping Keras MLP. "
            "Install with: pip install 'fakedetect[nn]'",
            file=sys.stderr,
        )
        return {}


def _find_data_dir() -> Path:
    """Best-effort search for the data directory."""
    candidates = [
        Path(__file__).parent.parent.parent / "data",
        Path.cwd() / "data",
    ]
    for p in candidates:
        if p.is_dir():
            return p
    return candidates[0]
