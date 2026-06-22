"""Feature engineering and preprocessing.

Key design decision
-------------------
The :class:`Preprocessor` fits its :class:`~sklearn.preprocessing.StandardScaler`
**on training data only** and then transforms both train and test sets using the
*same* fitted scaler.  The original notebook had a data-leakage bug where
``scaler.fit_transform(X_test)`` was called instead of ``scaler.transform(X_test)``,
meaning test statistics contaminated the scaling step.  That bug is fixed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from fakedetect.data import FEATURE_COLUMNS, TARGET


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features *X* and target *y*.

    Parameters
    ----------
    df:
        DataFrame that must contain :data:`~fakedetect.data.FEATURE_COLUMNS`
        and the ``fake`` target column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` where *X* has exactly the feature columns in schema order.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()
    return X, y


class Preprocessor:
    """Fit-on-train, transform-both scaler wrapper.

    Usage
    -----
    >>> prep = Preprocessor()
    >>> X_train_scaled, X_test_scaled = prep.fit_transform(X_train, X_test)

    The fitted :attr:`scaler` is accessible for inspection and persistence.
    """

    def __init__(self) -> None:
        self.scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    def fit_transform(
        self,
        X_train: pd.DataFrame | np.ndarray,
        X_test: pd.DataFrame | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit scaler on *X_train*, then transform both splits.

        Parameters
        ----------
        X_train:
            Training feature matrix.  The scaler is **fitted here only**.
        X_test:
            Test feature matrix.  Transformed with *train* statistics — no
            data leakage.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X_train_scaled, X_test_scaled)``
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Deliberately NOT re-fitting on test data (original leakage bug fixed)
        X_test_scaled = self.scaler.transform(X_test)
        self._fitted = True
        return X_train_scaled, X_test_scaled

    @property
    def train_mean_(self) -> np.ndarray:
        """Mean of each feature computed on training data."""
        self._require_fitted()
        return self.scaler.mean_  # type: ignore[return-value]

    @property
    def train_scale_(self) -> np.ndarray:
        """Std-dev of each feature computed on training data."""
        self._require_fitted()
        return self.scaler.scale_  # type: ignore[return-value]

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Preprocessor has not been fitted yet.  Call fit_transform first.")
