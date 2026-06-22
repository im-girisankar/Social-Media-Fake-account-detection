"""Model constructors.

Tested core (no TensorFlow required)
--------------------------------------
- :func:`build_logreg` — Logistic Regression baseline.
- :func:`build_random_forest` — Random Forest baseline.

Optional deep-learning model (TensorFlow/Keras)
-----------------------------------------------
- :func:`build_keras_mlp` — Feed-forward Dense MLP refactored from the
  original notebook.  **Requires tensorflow** (``pip install fakedetect[nn]``).
  Raises a clear :class:`ImportError` if TensorFlow is not installed.

Architecture note
-----------------
The original notebook was sometimes labelled "LSTM" in passing comments, but
the architecture is entirely feed-forward Dense layers — there is nothing
recurrent about it.  LSTMs suit sequential data; this dataset consists of
independent tabular profile features.  The name has been corrected throughout.

Fixes applied to the Keras model vs. the original
--------------------------------------------------
- Added :class:`~keras.callbacks.EarlyStopping` (patience=15, restore best
  weights) so the model stops when validation loss stops improving.
- Reduced ``max_epochs`` from 500 to 200 (EarlyStopping will usually trigger
  well before that).
- Learning rate kept at 0.01 (matches the original).
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------------
# sklearn baselines (no TF dependency)
# ---------------------------------------------------------------------------


def build_logreg(seed: int = 42) -> LogisticRegression:
    """Return a configured Logistic Regression classifier.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Unfitted estimator ready for ``.fit()``.
    """
    return LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )


def build_random_forest(seed: int = 42) -> RandomForestClassifier:
    """Return a configured Random Forest classifier.

    Parameters
    ----------
    seed:
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Unfitted estimator ready for ``.fit()``.
    """
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# Optional Keras MLP (lazy import — TF not required for core tests)
# ---------------------------------------------------------------------------


def build_keras_mlp(input_dim: int, seed: int = 42) -> Any:
    """Return a compiled Keras Sequential MLP for tabular binary classification.

    This is a cleaned-up version of the original notebook's architecture:
    Dense(50) → Dense(150) → Dropout(0.3) → Dense(25) → Dropout(0.3) → Dense(2, softmax).

    **Requires tensorflow** (``pip install fakedetect[nn]``).

    Parameters
    ----------
    input_dim:
        Number of input features (11 for the real dataset).
    seed:
        Random seed passed to TensorFlow for reproducibility.

    Returns
    -------
    keras.Sequential
        Compiled model ready for ``.fit()``.

    Raises
    ------
    ImportError
        If TensorFlow is not installed.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415
        from tensorflow import keras  # noqa: PLC0415
        from tensorflow.keras.layers import Dense, Dropout  # noqa: PLC0415
        from tensorflow.keras.models import Sequential  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for the Keras MLP model.  "
            "Install it with:  pip install 'fakedetect[nn]'"
        ) from exc

    tf.random.set_seed(seed)

    model = Sequential(
        [
            Dense(50, input_dim=input_dim, activation="relu"),
            Dense(150, activation="relu"),
            Dropout(0.3),
            Dense(25, activation="relu"),
            Dropout(0.3),
            Dense(2, activation="softmax"),
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=opt,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_keras_callbacks(patience: int = 15) -> list[Any]:
    """Return training callbacks for the Keras MLP.

    Includes :class:`~keras.callbacks.EarlyStopping` (restores best weights)
    to prevent the 500-epoch over-fitting seen in the original notebook.

    **Requires tensorflow.**

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for Keras callbacks.  "
            "Install it with:  pip install 'fakedetect[nn]'"
        ) from exc

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        )
    ]
