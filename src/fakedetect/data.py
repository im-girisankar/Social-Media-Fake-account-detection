"""Data loading and synthetic dataset generation.

Real dataset
------------
Download from Kaggle: "Instagram fake spammer genuine accounts"
https://www.kaggle.com/datasets/free4ever1/instagram-fake-spammer-genuine-accounts
Expected files: ``insta_train.csv`` and ``insta_test.csv`` (place in ``data/``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

FEATURE_COLUMNS: list[str] = [
    "profile pic",           # 0/1 — has profile picture
    "nums/length username",  # float ratio — digits in username / username length
    "fullname words",        # int — word count in full name
    "nums/length fullname",  # float ratio — digits in full name / full name length
    "name==username",        # 0/1 — full name equals username
    "description length",    # int — bio character count
    "external URL",          # 0/1 — has external URL in bio
    "private",               # 0/1 — account is private
    "#posts",                # int — number of posts
    "#followers",            # int — follower count
    "#follows",              # int — following count
]

TARGET: str = "fake"


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------


def load_csv(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the real Instagram dataset CSVs.

    Parameters
    ----------
    train_path:
        Path to ``insta_train.csv``.
    test_path:
        Path to ``insta_test.csv``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — each includes all feature columns plus the
        ``fake`` target column.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    _validate_schema(train_df, "train")
    _validate_schema(test_df, "test")
    return train_df, test_df


def _validate_schema(df: pd.DataFrame, name: str) -> None:
    missing = set(FEATURE_COLUMNS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"{name} DataFrame is missing columns: {missing}")


# ---------------------------------------------------------------------------
# Synthetic dataset (no real CSVs needed)
# ---------------------------------------------------------------------------


def generate_synthetic(
    n_train: int = 600,
    n_test: int = 200,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a deterministic synthetic dataset that mirrors the real schema.

    The fake/real signal is genuine (not random):

    - Fake accounts are more likely to have no profile picture.
    - Fake accounts tend to have higher ``nums/length username`` ratios.
    - Fake accounts have lower follower counts and fewer posts.
    - Real accounts have longer descriptions and more profile words.

    Parameters
    ----------
    n_train:
        Number of training rows.
    n_test:
        Number of test rows.
    seed:
        Random seed for full reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` with the same column schema as the real data.
    """
    rng = np.random.default_rng(seed)
    train_df = _make_df(n_train, rng)
    test_df = _make_df(n_test, rng)
    return train_df, test_df


def _make_df(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate *n* rows with a learnable fake signal."""
    # ~50 % fake, ~50 % real
    fake = rng.integers(0, 2, size=n).astype(int)

    profile_pic = np.where(fake == 1, rng.binomial(1, 0.15, n), rng.binomial(1, 0.85, n))

    # Fake: skewed towards higher digit ratio in username
    nums_len_username = np.where(
        fake == 1,
        rng.beta(2, 1, n),   # right-skewed
        rng.beta(1, 4, n),   # left-skewed
    ).astype(float)

    fullname_words = np.where(
        fake == 1,
        rng.integers(0, 2, n),
        rng.integers(1, 4, n),
    ).astype(int)

    nums_len_fullname = np.where(
        fake == 1,
        rng.beta(2, 2, n),
        rng.beta(1, 5, n),
    ).astype(float)

    name_eq_username = np.where(
        fake == 1, rng.binomial(1, 0.4, n), rng.binomial(1, 0.1, n)
    )

    description_length = np.where(
        fake == 1,
        rng.integers(0, 30, n),
        rng.integers(20, 150, n),
    ).astype(int)

    external_url = np.where(
        fake == 1, rng.binomial(1, 0.1, n), rng.binomial(1, 0.5, n)
    )

    private = np.where(
        fake == 1, rng.binomial(1, 0.2, n), rng.binomial(1, 0.45, n)
    )

    posts = np.where(
        fake == 1,
        rng.integers(0, 10, n),
        rng.integers(5, 200, n),
    ).astype(int)

    followers = np.where(
        fake == 1,
        rng.integers(0, 100, n),
        rng.integers(50, 5000, n),
    ).astype(int)

    follows = np.where(
        fake == 1,
        rng.integers(100, 2000, n),
        rng.integers(50, 800, n),
    ).astype(int)

    df = pd.DataFrame(
        {
            "profile pic": profile_pic,
            "nums/length username": nums_len_username,
            "fullname words": fullname_words,
            "nums/length fullname": nums_len_fullname,
            "name==username": name_eq_username,
            "description length": description_length,
            "external URL": external_url,
            "private": private,
            "#posts": posts,
            "#followers": followers,
            "#follows": follows,
            "fake": fake,
        }
    )
    return df
