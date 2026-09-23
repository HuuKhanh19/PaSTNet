"""Command-line interface for preparation, training, and reporting."""

import argparse

from pastnet.config import DATASETS


def main(argv=None):
    parser = argparse.ArgumentParser(description="PaSTNet molecular property prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train", "run", "summarize"):
        p = sub.add_parser(command)
        p.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
        p.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
        if command in ("prepare", "run"):
            p.add_argument("--raw-dir", default="data/raw")
            p.add_argument("--workers", type=int, default=1, help="Conformer generation worker processes")
        if command != "summarize":
            p.add_argument("--data-dir", default="data/processed")
        if command != "prepare":
            p.add_argument("--results-dir", default="results")
        if command in ("train", "run"):
            p.add_argument("--device", default="cuda", help="cuda, cuda:0, cuda:1, or cpu (smoke only)")
            p.add_argument("--smoke", action="store_true", help="Two FP32 epochs on eight rows per split; never aggregate as E0")
            p.add_argument("--resume", action="store_true", help="Resume at the last complete epoch or skip completed runs")
    args = parser.parse_args(argv)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Split seeds must be unique")
    if args.command in ("prepare", "run"):
        from pastnet.data.prepare import prepare
        for dataset in datasets:
            prepare(dataset, args.raw_dir, args.data_dir, args.seeds, args.workers)
    if args.command in ("train", "run"):
        from pastnet.training import train
        for dataset in datasets:
            for seed in args.seeds:
                train(dataset, seed, args.data_dir, args.results_dir, args.device, args.smoke, args.resume)
    if args.command == "summarize" or args.command == "run" and not args.smoke:
        from pastnet.reporting import summarize
        summarize(args.results_dir, datasets, args.seeds)


if __name__ == "__main__":
    main()
