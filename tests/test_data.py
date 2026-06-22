"""Tests for fakedetect.data — schema, shape, determinism."""

from __future__ import annotations

import pandas as pd

from fakedetect.data import (
    FEATURE_COLUMNS,
    TARGET,
    generate_synthetic,
)


class TestConstants:
    def test_feature_columns_count(self):
        """There are exactly 11 feature columns."""
        assert len(FEATURE_COLUMNS) == 11

    def test_target_name(self):
        assert TARGET == "fake"

    def test_feature_names_are_strings(self):
        assert all(isinstance(c, str) for c in FEATURE_COLUMNS)

    def test_expected_feature_names(self):
        expected = {
            "profile pic",
            "nums/length username",
            "fullname words",
            "nums/length fullname",
            "name==username",
            "description length",
            "external URL",
            "private",
            "#posts",
            "#followers",
            "#follows",
        }
        assert set(FEATURE_COLUMNS) == expected


class TestGenerateSynthetic:
    def test_returns_two_dataframes(self):
        train, test = generate_synthetic(n_train=50, n_test=20, seed=0)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_row_counts(self):
        train, test = generate_synthetic(n_train=120, n_test=40, seed=1)
        assert len(train) == 120
        assert len(test) == 40

    def test_column_schema_train(self):
        train, _ = generate_synthetic(n_train=50, n_test=10, seed=2)
        for col in FEATURE_COLUMNS + [TARGET]:
            assert col in train.columns, f"Missing column: {col!r}"

    def test_column_schema_test(self):
        _, test = generate_synthetic(n_train=50, n_test=20, seed=2)
        for col in FEATURE_COLUMNS + [TARGET]:
            assert col in test.columns, f"Missing column: {col!r}"

    def test_no_null_values(self):
        train, test = generate_synthetic(n_train=100, n_test=50, seed=3)
        assert not train.isnull().any().any(), "Train has null values"
        assert not test.isnull().any().any(), "Test has null values"

    def test_target_is_binary(self):
        train, test = generate_synthetic(n_train=200, n_test=80, seed=4)
        assert set(train[TARGET].unique()).issubset({0, 1})
        assert set(test[TARGET].unique()).issubset({0, 1})

    def test_deterministic_with_same_seed(self):
        train_a, test_a = generate_synthetic(n_train=60, n_test=20, seed=7)
        train_b, test_b = generate_synthetic(n_train=60, n_test=20, seed=7)
        pd.testing.assert_frame_equal(train_a, train_b)
        pd.testing.assert_frame_equal(test_a, test_b)

    def test_different_seeds_differ(self):
        train_a, _ = generate_synthetic(n_train=60, n_test=20, seed=7)
        train_b, _ = generate_synthetic(n_train=60, n_test=20, seed=99)
        # At least one column must differ
        assert not train_a.equals(train_b)

    def test_binary_columns_are_0_or_1(self):
        binary_cols = ["profile pic", "name==username", "external URL", "private"]
        train, _ = generate_synthetic(n_train=300, n_test=10, seed=5)
        for col in binary_cols:
            vals = set(train[col].unique())
            assert vals.issubset({0, 1}), f"{col!r} has non-binary values: {vals}"

    def test_numeric_ratio_columns_in_range(self):
        train, _ = generate_synthetic(n_train=300, n_test=10, seed=6)
        for col in ["nums/length username", "nums/length fullname"]:
            assert train[col].between(0.0, 1.0).all(), f"{col!r} out of [0,1]"

    def test_integer_count_columns_non_negative(self):
        train, _ = generate_synthetic(n_train=300, n_test=10, seed=6)
        for col in ["fullname words", "description length", "#posts", "#followers", "#follows"]:
            assert (train[col] >= 0).all(), f"{col!r} has negative values"

    def test_both_classes_present_in_large_sample(self):
        train, _ = generate_synthetic(n_train=500, n_test=10, seed=8)
        counts = train[TARGET].value_counts()
        assert 0 in counts.index and 1 in counts.index, "Both classes should appear"
