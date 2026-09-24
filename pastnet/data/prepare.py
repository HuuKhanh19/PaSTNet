"""Raw CSV -> pinned scaffold splits -> conformer cache -> usable splits."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
from pathlib import Path

import pandas as pd
from rdkit import Chem

from pastnet._vendor.schnet_gp_preprocessing import preprocess_dataframe
from pastnet._vendor.schnet_gp_splitter import random_scaffold_split, generate_scaffold
from pastnet.config import load_config, reference_data
from pastnet.data.cache import initialize_cache
from pastnet.data.conformers import ConformerConfig, ConformerGenerationError, generate_conformers
from pastnet.io import frame_digest, json_text, sha256, write_json, write_once

SPLITTER_SHA256 = "5297c60a3343c95b6cac98d3bfbc3c36965c5499472d410d0f63d5efb61335af"


def molecule_identity(smiles):
    canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True)
    return canonical, hashlib.sha1(canonical.encode()).hexdigest()[:16]


def split_raw(dataset, raw_dir, data_dir, seeds=(0, 1, 2)):
    cfg = load_config(dataset)
    ref = reference_data()[dataset]
    path = Path(raw_dir) / cfg["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Place {cfg['file']} in {Path(raw_dir).resolve()}; see README.md")
    if sha256(path) != ref["raw_sha256"]:
        raise ValueError(f"{path}: raw CSV differs from the released E0 input fingerprint")
    raw = pd.read_csv(path)
    frame = preprocess_dataframe(raw, cfg["smiles_column"], [cfg["target_column"]], cfg["task"])
    splits = {}
    for seed in seeds:
        parts = random_scaffold_split(frame, frame.smiles.values, random_seed=seed, dataframe=True)
        sets = [{generate_scaffold(s, include_chirality=True) for s in part.smiles} for part in parts]
        if any(sets[i] & sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("Scaffold leakage between partitions")
        for name, part in zip(("train", "valid", "test"), parts):
            part = part.reset_index(drop=True)
            expected = ref["splits"][str(seed)][name]
            if len(part) != expected["rows"] or frame_digest(part) != expected["ordered_rows_sha256"]:
                raise ValueError(f"{dataset}/{seed}/{name}: partition differs from reference; check dependency versions")
            target = Path(data_dir) / dataset / "random_scaffold" / f"seed_{seed}" / f"{name}.csv"
            write_once(target, part.to_csv(index=False, lineterminator="\n").encode())
            splits[seed, name] = part
    return frame, splits


def _generate(smiles):
    try:
        return generate_conformers(smiles, ConformerConfig())
    except ConformerGenerationError as exc:
        return None, exc.metadata


def prepare(dataset, raw_dir="data/raw", data_dir="data/processed", seeds=(0, 1, 2), workers=1):
    if workers < 1:
        raise ValueError("workers must be positive")
    frame, splits = split_raw(dataset, raw_dir, data_dir, seeds)
    directory = Path(data_dir) / dataset
    cache = initialize_cache(directory / "conformers_etkdg_mmff", ConformerConfig())
    identities = dict(molecule_identity(s) for s in frame.smiles)
    pending = []
    for canonical, mol_id in sorted(identities.items(), key=lambda pair: pair[1]):
        if cache.path(mol_id).exists():
            cache.recover_entry(mol_id) if mol_id not in cache.index else cache.load(mol_id, expected_smiles=canonical)
        elif cache.index.get(mol_id, {}).get("status") != "failed":
            pending.append(canonical)
    print(f"{dataset}: {len(pending)} conformer ensembles to generate; workers={workers}", flush=True)

    def consume(results):
        for index, (coords, metadata) in enumerate(results, 1):
            if coords is None:
                cache.record_failure(metadata)
            else:
                cache.store(coords, metadata)
            if index % 25 == 0 or index == len(pending):
                print(f"{dataset}: conformers {index}/{len(pending)}", flush=True)

    if workers == 1:
        consume(map(_generate, pending))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(_generate, pending, chunksize=1))
    failed = {mol_id for mol_id in identities.values() if cache.index[mol_id]["status"] == "failed"}
    expected_failed = set(reference_data()[dataset]["excluded_conformer_ids"])
    if failed != expected_failed:
        raise ValueError(f"Conformer failures differ from reference: extra={sorted(failed-expected_failed)}, "
                         f"missing={sorted(expected_failed-failed)}. Cache is retained for inspection.")
    report = dict(dataset=dataset, raw_sha256=sha256(Path(raw_dir) / load_config(dataset)["file"]),
                  split_source_sha256=SPLITTER_SHA256, generator=asdict(ConformerConfig()),
                  cache_fingerprint=cache.fingerprint, cache_index_sha256=sha256(cache.directory / "index.jsonl"),
                  excluded=[cache.index[key] for key in sorted(failed)],
                  filter_policy="remove_fixed_conformer_failures_after_split_without_reshuffling", splits={})
    for (seed, name), part in splits.items():
        retained = part.loc[[molecule_identity(s)[1] not in failed for s in part.smiles]]
        expected = reference_data()[dataset]["splits"][str(seed)][name]
        if frame_digest(retained) != expected["retained_ordered_rows_sha256"]:
            raise ValueError("Retained partition differs from reference")
        path = directory / "splits" / f"seed_{seed}" / f"{name}.csv"
        write_once(path, retained.to_csv(index=False, lineterminator="\n").encode())
        report["splits"].setdefault(str(seed), {})[name] = dict(
            original_rows=len(part), rows=len(retained), sha256=sha256(path))
    # Per-seed manifests let independently prepared seeds coexist.
    for seed in seeds:
        manifest = dict(report, splits={str(seed): report["splits"][str(seed)]})
        write_once(directory / "splits" / f"seed_{seed}" / "manifest.json", json_text(manifest).encode())
    print(f"{dataset}: prepared seeds {list(seeds)}; excluded {len(failed)} molecules", flush=True)
    return report
