"""Frozen E0 recipes and dataset metadata."""

import json
from importlib.resources import files

DATASETS = ("esol", "freesolv", "lipo", "bace", "bbbp", "clintox")


def load_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}; choose from {DATASETS}")
    return json.loads(files("pastnet").joinpath(f"configs/{dataset}.json").read_text())


def reference_data():
    return json.loads(files("pastnet").joinpath("reference/data.json").read_text())
