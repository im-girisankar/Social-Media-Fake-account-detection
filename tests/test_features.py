"""Tests for fakedetect.features — Preprocessor (leakage regression test)."""

from __future__ import annotations

import numpy as np
import pytest

from fakedetect.data import generate_synthetic
from fakedetect.features import Preprocessor, split_xy


class TestSplitXy:
    def test_returns_correct_shapes(self):
        train, _ = generate_synthetic(n_train=100, n_test=20, seed=10)
        X, y = split_xy(train)
        assert X.shape == (100, 11)
        assert y.shape == (100,)

    def test_y_column_is_target(self):
        train, _ = generate_synthetic(n_train=80, n_test=20, seed=11)
        _, y = split_xy(train)
        assert y.name == "fake"

    def test_x_does_not_contain_target(self):
        train, _ = generate_synthetic(n_train=80, n_test=20, seed=12)
        X, _ = split_xy(train)
        assert "fake" not in X.columns


class TestPreprocessor:
    def _get_xy(self, seed: int = 20):
        train, test = generate_synthetic(n_train=200, n_test=80, seed=seed)
        X_train, y_train = split_xy(train)
        X_test, y_test = split_xy(test)
        return X_train, y_train, X_test, y_test

    def test_output_shapes(self):
        X_train, _, X_test, _ = self._get_xy()
        prep = Preprocessor()
        Xtr, Xte = prep.fit_transform(X_train, X_test)
        assert Xtr.shape == X_train.shape
        assert Xte.shape == X_test.shape

    def test_train_mean_near_zero(self):
        """After StandardScaler, train column means should be ~0."""
        X_train, _, X_test, _ = self._get_xy()
        prep = Preprocessor()
        Xtr, _ = prep.fit_transform(X_train, X_test)
        col_means = Xtr.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_train_std_near_one(self):
        """After StandardScaler, train column std-devs should be ~1."""
        X_train, _, X_test, _ = self._get_xy()
        prep = Preprocessor()
        Xtr, _ = prep.fit_transform(X_train, X_test)
        col_stds = Xtr.std(axis=0)
        np.testing.assert_allclose(col_stds, 1.0, atol=1e-10)

    # ------------------------------------------------------------------
    # KEY REGRESSION TEST: no data leakage
    # ------------------------------------------------------------------

    def test_no_data_leakage_scaler_uses_train_statistics(self):
        """Regression test: the scaler is fitted on train and the *same* fitted
        scaler (with TRAIN statistics) is used to transform the test set.

        Strategy
        --------
        1. Fit a ``Preprocessor`` on train + test (correct, leakage-free).
        2. Manually compute what the test array *would* look like if the scaler
           were **incorrectly** re-fitted on test data.
        3. Assert the two are different (proving the scaler was not re-fitted).
        4. Assert that ``prep.train_mean_`` matches the train column means
           (not the test column means), confirming the scaler statistics come
           from training data.
        """
        X_train, _, X_test, _ = self._get_xy(seed=42)

        prep = Preprocessor()
        _, X_test_scaled = prep.fit_transform(X_train, X_test)

        # Manually simulate the BUGGY leakage path: refit on test
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        leaky_scaler = StandardScaler()
        X_test_leaky = leaky_scaler.fit_transform(X_test)

        # The leaky and correct transformations must differ because train ≠ test statistics
        assert not np.allclose(
            X_test_scaled, X_test_leaky
        ), "Leakage: test scaled with train stats is the same as refit on test — bug not fixed!"

        # The preprocessor's mean must equal the TRAIN column means, not the test column means
        X_train_np = X_train.to_numpy(dtype=float)
        X_test_np = X_test.to_numpy(dtype=float)
        train_col_means = X_train_np.mean(axis=0)
        test_col_means = X_test_np.mean(axis=0)

        np.testing.assert_allclose(
            prep.train_mean_,
            train_col_means,
            atol=1e-10,
            err_msg="Preprocessor mean_ does not match train data — scaler may have been refit on test!",
        )

        # Confirm the stored mean is NOT the test mean (i.e. it's genuinely different)
        assert not np.allclose(
            prep.train_mean_, test_col_means, atol=1e-6
        ), "Train and test means are suspiciously identical — test your synthetic generator."

    def test_test_transformed_with_train_statistics_directly(self):
        """Verify test-set values are derived from train statistics.

        Manually apply train scaler params and assert the result matches
        what Preprocessor.fit_transform returns for the test split.
        """
        X_train, _, X_test, _ = self._get_xy(seed=55)
        prep = Preprocessor()
        _, X_test_scaled = prep.fit_transform(X_train, X_test)

        # Reconstruct test scaling using saved train stats
        X_test_np = X_test.to_numpy(dtype=float)
        manually_scaled = (X_test_np - prep.train_mean_) / prep.train_scale_

        np.testing.assert_allclose(
            X_test_scaled,
            manually_scaled,
            atol=1e-10,
            err_msg="Test set was NOT scaled with train statistics.",
        )

    def test_unfitted_preprocessor_raises(self):
        prep = Preprocessor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            _ = prep.train_mean_
