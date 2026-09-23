"""E0 training, validation selection, exact epoch-boundary resume, and evaluation."""

from contextlib import nullcontext
import csv
import json
import logging
import math
from importlib.metadata import version
from pathlib import Path
import platform
import random
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from pastnet.config import load_config
from pastnet.data.cache import ConformerCache
from pastnet.data.dataset import MoleculeConfig, MoleculeDataset, load_splits
from pastnet.data.features import FeatureDataset, identity, smoke_records
from pastnet.geometry.relative import fit_training_bond_scale
from pastnet.io import sha256, write_json
from pastnet.model import PaSTNet


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def standardizer(targets):
    values = np.asarray(targets, dtype=np.float64)
    std = float(values.std(ddof=0))
    return dict(mean=float(values.mean()), std=std if std > 1e-12 else 1.0, count=len(values))


def metrics(prediction, target, task):
    prediction = prediction.detach().double().cpu()
    target = target.detach().double().cpu()
    if not prediction.shape == target.shape or not len(target) or not torch.isfinite(prediction).all():
        raise ValueError("Metrics require aligned finite predictions and labels")
    if task == "regression":
        error = prediction - target
        return dict(rmse=error.square().mean().sqrt().item(), mae=error.abs().mean().item())
    if set(target.tolist()) != {0.0, 1.0}:
        raise ValueError("ROC-AUC requires both classes; the partition cannot be evaluated")
    return dict(roc_auc=float(roc_auc_score(target.numpy(), prediction.numpy())),
                average_precision=float(average_precision_score(target.numpy(), prediction.numpy())),
                bce=torch.nn.functional.binary_cross_entropy_with_logits(prediction, target).item())


def forward_items(model, items, device, precision):
    labels = torch.stack([item[0] for item in items]).to(device)
    originals = [item[1] for item in items]
    molecules = [m.to(device) for m in originals]
    context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if precision == "bfloat16" else nullcontext()
    try:
        with context:
            prediction = model(molecules).float().reshape(-1)
        if prediction.shape != labels.shape or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite predictions or inconsistent batch shape")
        return prediction, labels
    finally:
        # Keep static geometry on CPU between batches; do not pin the entire
        # dataset's conformer graphs in GPU memory.
        for molecule in originals:
            molecule._device_cache.clear()


def objective(prediction, labels, task, scaler):
    if task == "regression":
        target = (labels - scaler["mean"]) / scaler["std"]
        return torch.nn.functional.mse_loss(prediction, target)
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Binary classification requires labels in {0, 1}")
    return torch.nn.functional.binary_cross_entropy_with_logits(prediction, labels)


@torch.no_grad()
def evaluate(model, loader, cfg, scaler, device):
    model.eval()
    predictions, targets = [], []
    for items in loader:
        for start in range(0, len(items), cfg["microbatch_size"]):
            prediction, labels = forward_items(model, items[start:start+cfg["microbatch_size"]],
                                                device, cfg["evaluation_precision"])
            if cfg["task"] == "regression":
                prediction = prediction.double() * scaler["std"] + scaler["mean"]
            predictions.append(prediction.double().cpu())
            targets.append(labels.double().cpu())
    prediction, target = torch.cat(predictions), torch.cat(targets)
    return metrics(prediction, target, cfg["task"]), prediction, target


def runtime_provenance():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    return dict(git_commit=commit, git_dirty=dirty, python=platform.python_version(),
                packages={name: version(name) for name in
                          ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "rdkit", "jarvis-tools")},
                cuda=torch.version.cuda)


def capture_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                loader=generator.get_state())


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    generator.set_state(state["loader"].cpu())


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train(dataset, seed=0, data_dir="data/processed", results_dir="results", device="cuda", smoke=False, resume=False):
    cfg = load_config(dataset)
    if seed not in cfg["split_seeds"]:
        raise ValueError("E0 uses split seeds 0, 1, and 2")
    device = torch.device(device)
    if not smoke and (device.type != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("The E0 recipe requires a BF16-capable CUDA GPU. Use --smoke --device cpu for a functional check.")
    if smoke:
        cfg.update(max_epochs=2, training_precision="float32", evaluation_precision="float32")
    torch.set_num_threads(cfg["num_threads"])
    seed_everything(cfg["model_seed"])
    directory = Path(data_dir) / dataset
    split_dir = directory / "splits" / f"seed_{seed}"
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("train", "valid", "test"):
        if sha256(split_dir / f"{name}.csv") != manifest["splits"][str(seed)][name]["sha256"]:
            raise ValueError(f"Prepared {name} split was modified; run preparation in a clean directory")
    cache = ConformerCache(directory / "conformers_etkdg_mmff")
    if cache.fingerprint != manifest["cache_fingerprint"] or sha256(cache.directory / "index.jsonl") != manifest["cache_index_sha256"]:
        raise ValueError("Conformer cache manifest/index changed after data preparation")
    splits = load_splits(MoleculeConfig(data_root=directory / "splits", split_seed=seed, include_test=False))
    records = {"train": splits.train.records, "valid": splits.val.records}
    if smoke:
        records = {key: smoke_records(value, cfg["task"] == "classification") for key, value in records.items()}
    scaler = standardizer([r.target for r in records["train"]]) if cfg["task"] == "regression" else dict(mean=0., std=1., count=0)
    bond_scale = fit_training_bond_scale(SimpleNamespace(train=records["train"]), cache)
    cfg.update(split_seed=seed, smoke_test=smoke, device=str(device))
    output = Path(results_dir) / dataset / f"seed_{seed}"
    if smoke:
        output = output / "smoke"
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"Run exists: {output}; use --resume or another --results-dir")
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(recipe=cfg, split_manifest_sha256=sha256(manifest_path), cache_fingerprint=cache.fingerprint)
    if resume and (output / "config.json").exists():
        previous = json.loads((output / "config.json").read_text())
        if previous["signature"] != signature:
            raise ValueError("Cannot resume with different data, device, or recipe")
        if (output / "metrics.json").exists():
            print(f"{dataset}/{seed}: already completed", flush=True)
            return json.loads((output / "metrics.json").read_text())
    logger = logging.getLogger(f"pastnet.{dataset}.{seed}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = [logging.StreamHandler(), logging.FileHandler(output / "run.log", encoding="utf-8")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logger.addHandler(handler)
    try:
        model = PaSTNet(bond_scale=bond_scale.value).to(device)
        metadata = dict(signature=signature, standardizer=scaler, bond_scale=bond_scale.metadata(),
                        parameter_count=sum(p.numel() for p in model.parameters()),
                        n_train=len(records["train"]), n_valid=len(records["valid"]),
                        provenance=runtime_provenance(),
                        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
        write_json(output / "config.json", metadata)
        datasets = {key: FeatureDataset(value, cache) for key, value in records.items()}
        generator = torch.Generator().manual_seed(cfg["model_seed"])
        validation_generator = (torch.Generator().manual_seed(cfg["evaluation_loader_seed"])
                                if cfg["evaluation_loader_seed"] is not None else None)
        train_loader = DataLoader(datasets["train"], batch_size=cfg["batch_size"], shuffle=True,
                                  generator=generator, collate_fn=identity, num_workers=0)
        valid_loader = DataLoader(datasets["valid"], batch_size=cfg["batch_size"],
                                  generator=validation_generator, collate_fn=identity, num_workers=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        best, best_epoch, start_epoch, history = (math.inf if cfg["task"] == "regression" else -math.inf), 0, 1, []
        if resume and (output / "last.pt").exists():
            checkpoint = torch.load(output / "last.pt", map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            best, best_epoch, start_epoch = checkpoint["best"], checkpoint["best_epoch"], checkpoint["epoch"]+1
            history = checkpoint["history"]
            # last.pt also holds the selected state so a mid-write interruption
            # cannot pair an older resume checkpoint with a newer best.pt.
            best_state = checkpoint["best_state"]
            save_checkpoint(output / "best.pt", best_state)
            restore_rng(checkpoint["rng"], generator)
            if validation_generator is not None:
                validation_generator.set_state(checkpoint["validation_rng"].cpu())
        elif resume and (output / "test_evaluation.json").exists():
            raise RuntimeError("Test evaluation already started; inspect its receipt before recovery")
        else:
            best_state = None
        logger.info("PaSTNet %s seed=%s parameters=%s train=%s valid=%s", dataset, seed,
                    metadata["parameter_count"], len(records["train"]), len(records["valid"]))
        began = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, cfg["max_epochs"]+1):
            if best_epoch and epoch - 1 - best_epoch >= cfg["patience"]:
                break
            model.train()
            loss_sum, count = 0., 0
            epoch_start = time.monotonic()
            for items in train_loader:
                optimizer.zero_grad(set_to_none=True)
                for start in range(0, len(items), cfg["microbatch_size"]):
                    micro = items[start:start+cfg["microbatch_size"]]
                    prediction, target = forward_items(model, micro, device, cfg["training_precision"])
                    loss = objective(prediction, target, cfg["task"], scaler)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite training loss")
                    (loss * (len(micro) / len(items))).backward()
                    loss_sum += loss.item() * len(micro)
                    count += len(micro)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                optimizer.step()
            val, _, _ = evaluate(model, valid_loader, cfg, scaler, device)
            score = val[cfg["metric"]]
            improved = score < best if cfg["task"] == "regression" else score > best
            if improved:
                best, best_epoch = score, epoch
                best_state = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                  epoch=epoch, val=val, config=metadata)
                save_checkpoint(output / "best.pt", best_state)
            row = dict(epoch=epoch, train_loss=loss_sum/count, **{f"val_{k}":v for k,v in val.items()},
                       lr=optimizer.param_groups[0]["lr"], seconds=time.monotonic()-epoch_start)
            history.append(row)
            save_checkpoint(output / "last.pt", dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            epoch=epoch, best=best, best_epoch=best_epoch, best_state=best_state,
                            rng=capture_rng(generator), validation_rng=validation_generator.get_state() if validation_generator else None,
                            history=history))
            with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            logger.info("epoch=%d loss=%.6f val_%s=%.6f best=%.6f", epoch, row["train_loss"], cfg["metric"], score, best)
        checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        # Report the validation measurement that selected this checkpoint.
        # CUDA scatter reductions with BF16 can vary on repeated evaluation;
        # a fresh validation pass must not redefine checkpoint selection.
        val = checkpoint["val"]
        receipt = output / "test_evaluation.json"
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(dict(status="started", best_epoch=best_epoch), handle)
        # Test features are instantiated only after checkpoint selection.
        test_splits = load_splits(MoleculeConfig(data_root=directory / "splits", split_seed=seed))
        test_records = test_splits.test.records
        if smoke:
            test_records = smoke_records(test_records, cfg["task"] == "classification")
        test_data = FeatureDataset(test_records, cache)
        test_loader = DataLoader(test_data, batch_size=cfg["batch_size"], collate_fn=identity,
                                generator=torch.Generator().manual_seed(42) if validation_generator is not None else None)
        test, predictions, labels = evaluate(model, test_loader, cfg, scaler, device)
        with (output / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["mol_id", "smiles", "target", "prediction" if cfg["task"] == "regression" else "logit"])
            writer.writerows((r.mol_id, r.smiles, float(y), float(p)) for r, y, p in zip(test_records, labels, predictions))
        result = dict(dataset=dataset, split_seed=seed, model_seed=42, task=cfg["task"], smoke_test=smoke,
                      best_epoch=best_epoch, epochs_completed=history[-1]["epoch"], val=val, test=test,
                      test_evaluations=1, n_train=len(records["train"]), n_valid=len(records["valid"]), n_test=len(test_records),
                      parameter_count=metadata["parameter_count"], elapsed_seconds=time.monotonic()-began,
                      peak_cuda_memory_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else 0.)
        write_json(receipt, dict(status="completed", best_epoch=best_epoch, metrics=test))
        write_json(output / "metrics.json", result)
        logger.info("Completed %s seed=%s test_%s=%.6f", dataset, seed, cfg["metric"], test[cfg["metric"]])
        return result
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
