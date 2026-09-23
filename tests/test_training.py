import json

import pytest
import torch

import pastnet.training as training
from pastnet.reporting import summarize


@pytest.mark.parametrize("dataset,metric", [("esol", "rmse"), ("bace", "roc_auc")])
def test_training_checkpoint_test_once(tiny_data, tmp_path, dataset, metric):
    result = training.train(dataset, data_dir=tiny_data, results_dir=tmp_path, device="cpu", smoke=True)
    assert result["test_evaluations"] == 1 and result["smoke_test"]
    assert metric in result["test"] and result["parameter_count"] == 667548
    output = tmp_path / dataset / "seed_0" / "smoke"
    before = (output / "test_predictions.csv").read_bytes()
    replay = training.train(dataset, data_dir=tiny_data, results_dir=tmp_path, device="cpu", smoke=True, resume=True)
    assert replay == result
    assert (output / "test_predictions.csv").read_bytes() == before
    with pytest.raises(FileExistsError):
        training.train(dataset, data_dir=tiny_data, results_dir=tmp_path, device="cpu", smoke=True)


def test_interrupted_epoch_resume_matches_uninterrupted(tiny_data, tmp_path, monkeypatch):
    training.train("esol", data_dir=tiny_data, results_dir=tmp_path / "full", device="cpu", smoke=True)
    save = training.save_checkpoint

    def interrupt(path, state):
        save(path, state)
        if path.name == "last.pt" and state["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    monkeypatch.setattr(training, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        training.train("esol", data_dir=tiny_data, results_dir=tmp_path / "resume", device="cpu", smoke=True)
    monkeypatch.setattr(training, "save_checkpoint", save)
    training.train("esol", data_dir=tiny_data, results_dir=tmp_path / "resume", device="cpu", smoke=True, resume=True)
    a = tmp_path / "full/esol/seed_0/smoke"
    b = tmp_path / "resume/esol/seed_0/smoke"
    assert (a / "test_predictions.csv").read_bytes() == (b / "test_predictions.csv").read_bytes()
    first = torch.load(a / "last.pt", weights_only=False)["model"]
    second = torch.load(b / "last.pt", weights_only=False)["model"]
    for key in first:
        torch.testing.assert_close(first[key], second[key], atol=0, rtol=0)


def test_summary_uses_sample_std_and_rejects_smoke(tmp_path):
    for seed, value in enumerate((1., 2., 3.)):
        path = tmp_path / "esol" / f"seed_{seed}"
        path.mkdir(parents=True)
        (path / "metrics.json").write_text(json.dumps(dict(dataset="esol", split_seed=seed,
            smoke_test=False, test_evaluations=1, test=dict(rmse=value))))
    rows = summarize(tmp_path, ("esol",))
    assert rows[0]["mean"] == 2. and rows[0]["std"] == 1.
    path = tmp_path / "esol/seed_0/metrics.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps(dict(record, smoke_test=True)))
    with pytest.raises(ValueError, match="debug"):
        summarize(tmp_path, ("esol",))
