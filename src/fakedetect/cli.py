"""Command-line interface for fakedetect.

Usage
-----
.. code-block:: bash

    # Run on synthetic data (no real CSVs or TensorFlow needed)
    python -m fakedetect.cli train --synthetic

    # Run on real data in the default data/ directory
    python -m fakedetect.cli train

    # Run on real data in a custom directory
    python -m fakedetect.cli train --data path/to/dir

    # Also train the optional Keras MLP (requires TensorFlow)
    python -m fakedetect.cli train --synthetic --nn
"""

from __future__ import annotations

import argparse
import sys

from fakedetect.train import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fakedetect",
        description="Instagram fake-account detection — reproducible ML pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", help="Train and evaluate models.")
    train_p.add_argument(
        "--data",
        metavar="DIR",
        default=None,
        help="Directory containing insta_train.csv and insta_test.csv. "
        "Defaults to data/ relative to the project root.",
    )
    train_p.add_argument(
        "--synthetic",
        action="store_true",
        help="Use a synthetic dataset instead of real CSVs.",
    )
    train_p.add_argument(
        "--nn",
        action="store_true",
        help="Also train the Keras MLP (requires TensorFlow: pip install fakedetect[nn]).",
    )
    train_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed (default: 42).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``fakedetect`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "train":
        run_pipeline(
            data_dir=args.data,
            use_synthetic=args.synthetic,
            train_nn=args.nn,
            seed=args.seed,
        )
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
