"""Aggregate completed seeds using the paper's sample-standard-deviation convention."""

import csv
import json
from pathlib import Path

import numpy as np

from pastnet.config import DATASETS, load_config
from pastnet.io import write_json


def summarize(results_dir="results", datasets=DATASETS, seeds=(0, 1, 2)):
    rows = []
    for dataset in datasets:
        metric = load_config(dataset)["metric"]
        values = []
        for seed in seeds:
            path = Path(results_dir) / dataset / f"seed_{seed}" / "metrics.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed E0 run: {path}")
            result = json.loads(path.read_text())
            if result["smoke_test"] or result["test_evaluations"] != 1:
                raise ValueError(f"Cannot report incomplete/debug metrics: {path}")
            if result["dataset"] != dataset or result["split_seed"] != seed:
                raise ValueError(f"Result identity mismatch: {path}")
            values.append(result["test"][metric])
        rows.append(dict(dataset=dataset, metric=metric, seeds=list(seeds), n=len(values),
                         mean=float(np.mean(values)), std=float(np.std(values, ddof=1)) if len(values)>1 else None,
                         per_seed=values))
    write_json(Path(results_dir) / "summary.json", rows)
    with (Path(results_dir) / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
    for row in rows:
        std = f"{row['std']:.4f}" if row["std"] is not None else "n/a"
        print(f"{row['dataset']:<9} {row['metric']:<8} {row['mean']:.4f} +/- {std} (n={row['n']})")
    return rows
