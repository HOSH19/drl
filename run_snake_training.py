#!/usr/bin/env python3
"""Project root entry point; delegates to :mod:`experiments.train_snake`."""

import sys
from pathlib import Path


def main() -> None:
    """Parse CLI and run training or watch mode (see ``train_snake``)."""
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root / "src"))
    from experiments.train_snake import main as train_main

    train_main()


if __name__ == "__main__":
    main()
