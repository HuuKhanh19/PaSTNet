import json
import shutil

import pytest
import torch

from pastnet.config import reference_data
from pastnet.io import write_json
from pastnet.table2 import DATASETS, SEEDS, VARIANTS, build_model, load_recipe, main, summarize
from pastnet.training import train


@pytest.mark.parametrize("variant", VARIANTS)
def test_capacity_shared_initialization_and_rng(variant):
    torch.manual_seed(42)
    reference = build_model("full_past", bond_scale=.065)
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = build_model(variant, bond_scale=.065)
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.parameter_counts()["total"] == VARIANTS[variant][3]
    for name, value in model.state_dict().items():
        if ".past." not in name:
            torch.testing.assert_close(value, reference.state_dict()[name], atol=0, rtol=0)


@pytest.mark.parametrize("dataset", ("esol", "bace"))
@pytest.mark.parametrize("variant", VARIANTS)
def test_operator_training_and_resume(tiny_data, tmp_path, dataset, variant):
    result = train(dataset, data_dir=tiny_data, results_dir=tmp_path, device="cpu",
                   smoke=True, variant=variant)
    assert result["parameter_count"] == VARIANTS[variant][3]
    assert result["variant"] == variant and result["protocol"] == "table2"
    assert result["test_evaluations"] == 1 and result["smoke_test"]
    output = tmp_path / "table2" / variant / dataset / "seed_0" / "smoke"
    predictions = (output / "test_predictions.csv").read_bytes()
    metadata = json.loads((output / "config.json").read_text())
    recipe = metadata["signature"]["recipe"]
    assert recipe["variant"] == variant and recipe["max_conformers"] == 5
    assert metadata["standardizer"]["count"] == (8 if dataset == "esol" else 0)
    assert train(dataset, data_dir=tiny_data, results_dir=tmp_path, device="cpu",
                 smoke=True, resume=True, variant=variant) == result
    assert (output / "test_predictions.csv").read_bytes() == predictions


def test_fifth_split_is_trainable(tiny_data, tmp_path):
    source = tiny_data / "esol/splits/seed_0"
    target = tiny_data / "esol/splits/seed_4"
    shutil.copytree(source, target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["splits"]["4"] = manifest["splits"].pop("0")
    write_json(target / "manifest.json", manifest)
    result = train("esol", seed=4, data_dir=tiny_data, results_dir=tmp_path,
                   device="cpu", smoke=True, variant="no_past")
    assert result["split_seed"] == 4


def test_table2_recipes_and_cli_matrix(monkeypatch):
    ref = reference_data()
    for dataset in DATASETS:
        recipes = []
        for variant in VARIANTS:
            cfg = load_recipe(dataset, variant)
            assert cfg.pop("split_seeds") == list(SEEDS)
            cfg.pop("variant")
            cfg.pop("expected_parameters")
            recipes.append(cfg)
        assert all(cfg == recipes[0] for cfg in recipes)
        assert all(str(seed) in ref[dataset]["splits"] for seed in SEEDS)
    calls = []
    monkeypatch.setattr("pastnet.training.train", lambda dataset, seed, *a, **kw: calls.append((dataset, seed, kw["variant"])))
    main(["train"])
    assert set(calls) == {(d, s, v) for d in DATASETS for s in SEEDS for v in VARIANTS}
    assert len(calls) == 120
    with pytest.raises(SystemExit):
        main(["train", "--seeds", "0", "0"])
    with pytest.raises(ValueError):
        load_recipe("lipo", "full_past")


def test_table2_summary_scale_std_and_identity(tmp_path):
    for seed, value in enumerate((.7, .8, .9)):
        path = tmp_path / "table2/no_past/bace" / f"seed_{seed}" / "metrics.json"
        write_json(path, dict(protocol="table2", variant="no_past", dataset="bace", split_seed=seed,
                             model_seed=42, parameter_count=410892, smoke_test=False,
                             test_evaluations=1, test=dict(roc_auc=value)))
    rows = summarize(tmp_path, ("bace",), ("no_past",), (0, 1, 2))
    assert rows[0]["mean"] == pytest.approx(80.)
    assert rows[0]["std"] == pytest.approx(10.)
    assert rows[0]["metric"] == "roc_auc_percent"
    record = json.loads(path.read_text())
    write_json(path, dict(record, variant="full_past"))
    with pytest.raises(ValueError, match="Mismatched"):
        summarize(tmp_path, ("bace",), ("no_past",), (0, 1, 2))
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path, ("bace",), ("no_past",), (4,))
