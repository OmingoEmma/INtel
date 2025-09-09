from __future__ import annotations
import os
from typing import Any, Dict
import yaml

DEFAULT_PATH = "configs/config.yaml"

def load_config(path: str = DEFAULT_PATH) -> Dict[str, Any]:
    """
    Load a YAML config file and return a dict.
    Defaults to configs/config.yaml relative to repo root.
    """
    # resolve relative to repo root
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
