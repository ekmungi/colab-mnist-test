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


def write_text_spec(path, text):
    """Writes a specification file from its full text and returns the path as a string."""
    path.write_text(text)
    return str(path)


def test_a_list_of_data_tables_with_one_mode_uses_that_mode(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "store"\n[train]\nepochs = 1\n')
    assert train.parse_args(["--spec", spec]).data_mode == "store"


def test_a_single_data_table_in_a_list_works_and_a_missing_mode_means_fetch(tmp_path):
    only_store = write_text_spec(tmp_path / "a.toml", '[[data]]\nname = "A"\nmode = "store"\n')
    no_mode = write_text_spec(tmp_path / "b.toml", '[[data]]\nname = "A"\n[[data]]\nname = "B"\n')
    assert train.parse_args(["--spec", only_store]).data_mode == "store"
    assert train.parse_args(["--spec", no_mode]).data_mode == "fetch"


def test_data_tables_with_different_modes_stop_with_a_message_that_names_the_choice(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "fetch"\n')
    with pytest.raises(SystemExit) as error:
        train.parse_args(["--spec", spec])
    message = str(error.value)
    assert "--data-mode" in message and "store" in message and "fetch" in message and "A" in message and "B" in message


def test_different_modes_are_fine_when_the_command_line_chooses_one(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[[data]]\nname = "A"\nmode = "store"\n[[data]]\nname = "B"\nmode = "fetch"\n')
    assert train.parse_args(["--spec", spec, "--data-mode", "store"]).data_mode == "store"


def test_a_train_key_that_names_a_fixed_option_becomes_its_default(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml",
                           '[train]\nmodel = "mlp"\naugment = true\nexp = "exp-007"\nmax_steps = 5\nepochs = 2\n')
    args = train.parse_args(["--spec", spec])
    assert (args.model, args.augment, args.exp, args.max_steps, args.epochs) == ("mlp", True, "exp-007", 5, 2)


def test_the_command_line_still_overrides_a_colliding_train_key(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmodel = "mlp"\nexp = "exp-007"\n')
    args = train.parse_args(["--spec", spec, "--model", "cnn", "--exp", "exp-002"])
    assert (args.model, args.exp) == ("cnn", "exp-002")


def test_a_train_key_that_names_a_model_that_does_not_exist_is_rejected_clearly(tmp_path):
    spec = write_text_spec(tmp_path / "spec.toml", '[train]\nmodel = "transformer-xl"\n')
    with pytest.raises(SystemExit):
        train.parse_args(["--spec", spec])
