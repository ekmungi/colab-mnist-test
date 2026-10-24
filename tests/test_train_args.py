"""Tests for how train.py reads the data mode from the experiment specification and the command line."""
import pytest

pytest.importorskip("torch")

import train  # noqa: E402


def write_spec(path, data_table):
    """Writes a minimal experiment specification with the given [data] table text."""
    path.write_text(f"[data]\n{data_table}\n[train]\nepochs = 1\n")
    return str(path)


def test_the_data_mode_defaults_to_fetch():
    assert train.parse_args([]).data_mode == "fetch"


def test_the_data_mode_comes_from_the_specification(tmp_path):
    spec = write_spec(tmp_path / "spec.toml", 'mode = "store"')
    assert train.parse_args(["--spec", spec]).data_mode == "store"


def test_the_command_line_overrides_the_specification(tmp_path):
    spec = write_spec(tmp_path / "spec.toml", 'mode = "store"')
    assert train.parse_args(["--spec", spec, "--data-mode", "fetch"]).data_mode == "fetch"


def test_an_unknown_data_mode_is_rejected():
    with pytest.raises(SystemExit):
        train.parse_args(["--data-mode", "sometimes"])
