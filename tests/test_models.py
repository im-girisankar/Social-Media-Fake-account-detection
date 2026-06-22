"""Tests for fakedetect.models — sklearn baselines and optional TF guard."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fakedetect.data import generate_synthetic
from fakedetect.features import Preprocessor, split_xy
from fakedetect.models import build_logreg, build_random_forest


@pytest.fixture(scope="module")
def prepared_data():
    """Shared prepared data for model tests."""
    train, test = generate_synthetic(n_train=400, n_test=100, seed=99)
    X_train, y_train = split_xy(train)
    X_test, y_test = split_xy(test)
    prep = Preprocessor()
    X_train_s, X_test_s = prep.fit_transform(X_train, X_test)
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    return X_train_s, y_train_np, X_test_s, y_test_np


class TestBuildLogreg:
    def test_returns_logreg_instance(self):
        assert isinstance(build_logreg(), LogisticRegression)

    def test_seed_is_set(self):
        lr = build_logreg(seed=7)
        assert lr.random_state == 7

    def test_fits_and_predicts(self, prepared_data):
        X_tr, y_tr, X_te, _ = prepared_data
        lr = build_logreg(seed=42)
        lr.fit(X_tr, y_tr)
        preds = lr.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert set(preds).issubset({0, 1})

    def test_accuracy_above_chance(self, prepared_data):
        """Logreg must beat random guessing (>60 %) on a learnable dataset."""
        X_tr, y_tr, X_te, y_te = prepared_data
        lr = build_logreg(seed=42)
        lr.fit(X_tr, y_tr)
        acc = (lr.predict(X_te) == y_te).mean()
        assert acc > 0.60, f"Logreg accuracy too low: {acc:.3f}"

    def test_predict_proba_available(self, prepared_data):
        X_tr, y_tr, X_te, _ = prepared_data
        lr = build_logreg(seed=42)
        lr.fit(X_tr, y_tr)
        proba = lr.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestBuildRandomForest:
    def test_returns_rf_instance(self):
        assert isinstance(build_random_forest(), RandomForestClassifier)

    def test_seed_is_set(self):
        rf = build_random_forest(seed=13)
        assert rf.random_state == 13

    def test_fits_and_predicts(self, prepared_data):
        X_tr, y_tr, X_te, _ = prepared_data
        rf = build_random_forest(seed=42)
        rf.fit(X_tr, y_tr)
        preds = rf.predict(X_te)
        assert preds.shape == (len(X_te),)
        assert set(preds).issubset({0, 1})

    def test_accuracy_above_chance(self, prepared_data):
        """Random Forest must beat random guessing (>60 %) on a learnable dataset."""
        X_tr, y_tr, X_te, y_te = prepared_data
        rf = build_random_forest(seed=42)
        rf.fit(X_tr, y_tr)
        acc = (rf.predict(X_te) == y_te).mean()
        assert acc > 0.60, f"RF accuracy too low: {acc:.3f}"

    def test_feature_importances(self, prepared_data):
        X_tr, y_tr, _, _ = prepared_data
        rf = build_random_forest(seed=42)
        rf.fit(X_tr, y_tr)
        assert len(rf.feature_importances_) == 11
        np.testing.assert_allclose(rf.feature_importances_.sum(), 1.0, atol=1e-6)


class TestKerasMlpGuard:
    def test_raises_import_error_when_tf_absent(self, monkeypatch):
        """build_keras_mlp must raise ImportError with a helpful message if TF is not installed."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorflow" or name.startswith("tensorflow."):
                raise ImportError("No module named 'tensorflow'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        # Need to reload models to pick up monkeypatched import
        import fakedetect.models as models_mod  # noqa: PLC0415

        with pytest.raises(ImportError, match="TensorFlow is required"):
            models_mod.build_keras_mlp(input_dim=11)
