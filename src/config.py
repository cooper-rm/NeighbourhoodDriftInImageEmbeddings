# src/config.py

from pathlib import Path
import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries.
    Values in `override` take precedence over `base`.
    """
    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(*paths) -> dict:
    """
    Load and merge multiple YAML config files.

    Later files override earlier ones.
    """
    config = {}

    for path in paths:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            cfg_part = yaml.safe_load(f)

        if cfg_part is None:
            continue

        if not isinstance(cfg_part, dict):
            raise ValueError(f"Config file must contain a YAML dict: {path}")

        config = deep_merge(config, cfg_part)

    return config
