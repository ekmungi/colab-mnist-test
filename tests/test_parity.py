"""Parity tests: the script path (train.run) behaves like calling the shared fit() directly, and resumes exactly."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import models  # noqa: E402
import quicktrain  # noqa: E402
import trainer  # noqa: E402
import train  # noqa: E402
from synthetic import make_dataset  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def quiet(message):
    """Swallows the log lines so test output stays clean."""


def args_for(tmp_path, run_id, *extra):
    return train.parse_args(["--out-dir", str(tmp_path), "--run-id", run_id, "--model", "mlp", "--epochs", "3",
                             "--batch-size", "16", "--checkpoint-every", "5", "--seed", "0", *extra])


def test_the_script_path_matches_a_direct_call_to_fit(tmp_path):
    data = make_dataset(64)
    metrics = train.run(args_for(tmp_path, "script"), data, data, data, say=quiet)

    train.set_seeds(0)
    model = models.build_model("mlp")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _, state, _ = trainer.fit(model, optimizer, data, data, 3, 16, "cpu", seed=0, eval_train=False)
    expected = quicktrain.accuracy(model, DataLoader(data, batch_size=1000), "cpu")

    assert metrics["steps"] == state.step == 12
    assert metrics["val_accuracy"] == expected


def test_a_script_run_resumed_after_a_stop_gives_the_same_result_as_an_uninterrupted_run(tmp_path):
    data = make_dataset(64)
    whole = train.run(args_for(tmp_path, "whole"), data, data, data, quiet)

    first = train.run(args_for(tmp_path, "split", "--max-steps", "7"), data, data, data, quiet)
    assert first["stopped_early"] and first["steps"] == 7
    second = train.run(args_for(tmp_path, "split"), data, data, data, quiet)

    assert second["resumed_from"] == "checkpoint-00000005.pt"
    assert second["steps"] == whole["steps"] == 12
    assert (second["val_accuracy"], second["test_accuracy"]) == (whole["val_accuracy"], whole["test_accuracy"])
