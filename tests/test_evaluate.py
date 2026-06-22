"""Tests for fakedetect.evaluate — metric correctness and key presence."""

from __future__ import annotations

import numpy as np
import pytest

from fakedetect.evaluate import evaluate, format_report


class TestEvaluate:
    def _perfect(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        return y_true, y_pred

    def _random(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_pred = rng.integers(0, 2, 200)
        return y_true, y_pred

    # --- Required keys ---

    def test_returns_required_keys(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        required = {"accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"}
        assert required.issubset(result.keys())

    # --- Perfect predictions ---

    def test_perfect_accuracy(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_precision(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        assert result["precision"] == pytest.approx(1.0)

    def test_perfect_recall(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        assert result["recall"] == pytest.approx(1.0)

    def test_perfect_f1(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        assert result["f1"] == pytest.approx(1.0)

    # --- roc_auc ---

    def test_roc_auc_none_when_no_proba(self):
        y_true, y_pred = self._perfect()
        result = evaluate(y_true, y_pred)
        assert result["roc_auc"] is None

    def test_roc_auc_computed_when_proba_given(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        result = evaluate(y_true, y_pred, y_proba)
        assert result["roc_auc"] is not None
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_perfect_roc_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        result = evaluate(y_true, y_pred, y_proba)
        assert result["roc_auc"] == pytest.approx(1.0)

    # --- Value ranges ---

    def test_accuracy_in_range(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_precision_in_range(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert 0.0 <= result["precision"] <= 1.0

    def test_recall_in_range(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert 0.0 <= result["recall"] <= 1.0

    def test_f1_in_range(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert 0.0 <= result["f1"] <= 1.0

    # --- Confusion matrix ---

    def test_confusion_matrix_shape(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert result["confusion_matrix"].shape == (2, 2)

    def test_confusion_matrix_sum_equals_n(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert result["confusion_matrix"].sum() == len(y_true)

    def test_confusion_matrix_non_negative(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        assert (result["confusion_matrix"] >= 0).all()

    # --- All-wrong predictions ---

    def test_all_wrong_accuracy_near_zero(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 0, 0, 0])  # completely wrong
        result = evaluate(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(0.0)

    # --- Metrics are floats ---

    def test_metrics_are_floats(self):
        y_true, y_pred = self._random()
        result = evaluate(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert isinstance(result[key], float), f"{key} should be float"


class TestFormatReport:
    def test_contains_model_name(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = evaluate(y_true, y_pred)
        report = format_report("My Model", metrics)
        assert "My Model" in report

    def test_contains_all_metric_labels(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        metrics = evaluate(y_true, y_pred)
        report = format_report("Test", metrics)
        for label in ("Accuracy", "Precision", "Recall", "F1"):
            assert label in report

    def test_returns_string(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        metrics = evaluate(y_true, y_pred)
        assert isinstance(format_report("X", metrics), str)
