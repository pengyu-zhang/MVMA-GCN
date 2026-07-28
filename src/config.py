"""YAML configuration loading with per-dataset overrides."""

import copy

import yaml


def _deep_update(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, dataset):
    """Load a YAML config and merge the per-dataset override section.

    The returned dict has the dataset-specific values merged into the
    top-level sections, plus ``dataset`` set to the dataset name.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = cfg.pop("datasets", {})
    if dataset not in overrides:
        raise ValueError(
            f"dataset '{dataset}' not defined in {path} (available: {sorted(overrides)})"
        )
    cfg = _deep_update(copy.deepcopy(cfg), overrides[dataset])
    cfg["dataset"] = dataset
    cfg["config_path"] = path
    return cfg
