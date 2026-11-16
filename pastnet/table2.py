"""Run the eight controlled Table 2 operators with a shared training recipe."""

import argparse
import csv
import importlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from pastnet.config import load_config
from pastnet.io import write_json

DATASETS = ("esol", "freesolv", "bace")
SEEDS = (0, 1, 2, 3, 4)
# label, module, class, total trainable parameters
VARIANTS = {
    "no_past": ("Identity (no PaST)", "pastnet_no_past", "PaSTNetNoPaST", 410892),
    "same_path_attention": ("Same-path self-attention", "pastnet_ablation", "PaSTNetAblation", 667710),
    "centered_attention": ("Singleton-centered attention", "pastnet_centered_attention", "PaSTNetCenteredAttention", 667710),
    "no_pair_context": ("Pair-local PaST", "pastnet_no_pair_context", "PaSTNetNoPairContext", 618396),
    "pair_local_matched": ("Capacity-matched pair-local PaST", "pastnet_pair_local_matched", "PaSTNetPairLocalMatched", 667548),
    "no_singleton_subtraction": ("PaST without singleton subtraction", "pastnet_no_singleton_subtraction", "PaSTNetNoSingletonSubtraction", 667548),
    "ign15": ("Full 15-map equivariant tensor", "pastnet_ign15", "PaSTNetIGN15", 666972),
    "full_past": ("Full PaST", "network", "PaSTNet", 667548),
}


def load_recipe(dataset, variant):
    if dataset not in DATASETS or variant not in VARIANTS:
        raise ValueError("Unknown Table 2 dataset or operator")
    cfg = load_config(dataset)
    cfg.update(protocol="table2", variant=variant, split_seeds=list(SEEDS), max_conformers=5,
               expected_parameters=VARIANTS[variant][3])
    return cfg


def build_model(variant, *, bond_scale):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown Table 2 operator: {variant}")
    _, module, name, expected = VARIANTS[variant]
    constructor = getattr(importlib.import_module(f"pastnet.model.{module}"), name)
    options = dict(bond_scale=bond_scale)
    if variant == "same_path_attention":
        options["variant"] = variant
    model = constructor(**options)
    count = sum(p.numel() for p in model.parameters())
    if count != expected:
        raise ValueError(f"{variant}: expected {expected} trainable parameters, got {count}")
    return model


def summarize(results_dir="results", datasets=DATASETS, variants=tuple(VARIANTS), seeds=SEEDS):
    if (not datasets or not variants or not seeds or len(set(seeds)) != len(seeds)
            or any(s not in SEEDS for s in seeds)):
        raise ValueError("Choose nonempty datasets/operators and unique split seeds from 0 through 4")
    rows = []
    root = Path(results_dir) / "table2"
    for variant in variants:
        for dataset in datasets:
            cfg = load_recipe(dataset, variant)
            metric = cfg["metric"]
            values = []
            for seed in seeds:
                path = root / variant / dataset / f"seed_{seed}" / "metrics.json"
                result = json.loads(path.read_text(encoding="utf-8"))
                expected = dict(protocol="table2", variant=variant, dataset=dataset, split_seed=seed,
                                model_seed=42, parameter_count=VARIANTS[variant][3],
                                smoke_test=False, test_evaluations=1)
                if any(result.get(k) != v for k, v in expected.items()):
                    raise ValueError(f"Mismatched or debug Table 2 result: {path}")
                value = result["test"][metric]
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite metric: {path}")
                # Table 2 reports ROC-AUC in percent; raw run files stay on [0, 1].
                values.append(value * 100 if metric == "roc_auc" else value)
            rows.append(dict(variant=variant, label=VARIANTS[variant][0], dataset=dataset,
                             parameters=VARIANTS[variant][3],
                             metric="roc_auc_percent" if metric == "roc_auc" else metric,
                             seeds=list(seeds), n=len(values), mean=mean(values),
                             std=stdev(values) if len(values) > 1 else None, per_seed=values))
    write_json(root / "summary.json", rows)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["variant", "dataset", "parameters", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in writer.fieldnames} for r in rows)
    for row in rows:
        spread = f"{row['std']:.4f}" if row["std"] is not None else "n/a"
        print(f"{row['variant']} / {row['dataset']}: {row['mean']:.4f} +/- {spread} {row['metric']} (n={row['n']})")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "run", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        p.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=list(SEEDS))
        if command in ("prepare", "run"):
            p.add_argument("--raw-dir", default="data/raw")
            p.add_argument("--workers", type=int, default=1)
        if command != "summarize":
            p.add_argument("--data-dir", default="data/processed")
        if command != "prepare":
            p.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
            p.add_argument("--results-dir", default="results")
        if command in ("train", "run"):
            p.add_argument("--device", default="cuda")
            p.add_argument("--smoke", action="store_true", help="Two FP32 epochs on a small subset; excluded from summaries")
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Split seeds must be unique")
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if args.command in ("prepare", "run"):
        from pastnet.data.prepare import prepare
        for dataset in datasets:
            prepare(dataset, args.raw_dir, args.data_dir, args.seeds, args.workers)
    if args.command == "prepare":
        return
    variants = tuple(VARIANTS) if args.variant == "all" else (args.variant,)
    if args.command in ("train", "run"):
        from pastnet.training import train
        for dataset in datasets:
            for seed in args.seeds:
                for variant in variants:
                    train(dataset, seed, args.data_dir, args.results_dir, args.device,
                          args.smoke, args.resume, variant=variant)
    if args.command == "summarize" or args.command == "run" and not args.smoke:
        summarize(args.results_dir, datasets, variants, args.seeds)


if __name__ == "__main__":
    main()
