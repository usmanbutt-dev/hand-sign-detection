"""Utility: load the central config.yaml."""
from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    """Load configs/config.yaml and return as a dict.

    Args:
        path: Optional explicit path to a config file.
              Defaults to <project_root>/configs/config.yaml.

    Returns:
        Parsed configuration dictionary.
    """
    if path is None:
        # Resolve relative to this file: src/utils/ → project root
        path = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"

    with open(path, "r") as f:
        return yaml.safe_load(f)
