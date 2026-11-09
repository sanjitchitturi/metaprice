"""Configuration loading utilities."""

from __future__ import annotations

from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file and return as nested dict."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg
