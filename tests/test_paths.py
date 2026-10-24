"""Tests for paths.py: where the artifacts folder is, how the synced Drive project is found, and the folder naming rules."""
from pathlib import Path

import pytest

import paths


def make_mount(tmp_path, *relative_dirs):
    for relative in relative_dirs:
        (tmp_path / relative).mkdir(parents=True)
    return tmp_path


def test_the_environment_variable_wins():
    assert paths.artifact_root({"ARTIFACT_ROOT": "/somewhere"}, on_colab=True) == Path("/somewhere")


def test_drive_project_is_found_under_mydrive_research(tmp_path):
    mount = make_mount(tmp_path, f"MyDrive/research/{paths.PROJECT}")
    assert paths.drive_project_dir({}, mount) == mount / "MyDrive" / "research" / paths.PROJECT


def test_drive_project_is_found_in_a_computer_folder_synced_by_the_drive_app(tmp_path):
    mount = make_mount(tmp_path, f"Othercomputers/MY-PC/research/{paths.PROJECT}")
    assert paths.drive_project_dir({}, mount) == mount / "Othercomputers" / "MY-PC" / "research" / paths.PROJECT


def test_drive_project_environment_variable_overrides_the_search(tmp_path):
    assert paths.drive_project_dir({"DRIVE_PROJECT_DIR": "/x/y"}, tmp_path) == Path("/x/y")


def test_drive_project_is_none_when_nothing_matches(tmp_path):
    assert paths.drive_project_dir({}, tmp_path) is None


def test_on_colab_with_the_project_synced_artifacts_is_inside_it(tmp_path):
    mount = make_mount(tmp_path, f"Othercomputers/MY-PC/research/{paths.PROJECT}")
    assert paths.artifact_root({}, on_colab=True, mount=mount) == mount / "Othercomputers" / "MY-PC" / "research" / paths.PROJECT / "artifacts"


def test_on_colab_without_the_project_the_machine_disk_is_used(tmp_path):
    assert paths.artifact_root({}, on_colab=True, mount=tmp_path) == Path("/content/artifacts")


def test_on_this_computer_artifacts_sits_in_the_project_folder():
    assert paths.artifact_root({}, on_colab=False) == paths.project_root() / "artifacts"


def test_run_dir_builds_the_expected_path(tmp_path):
    assert paths.run_dir("exp-002", "run-007", tmp_path) == tmp_path / "exp-002" / "run-007"
    assert paths.run_dir("scratch", "smoke-001", tmp_path) == tmp_path / "scratch" / "smoke-001"


@pytest.mark.parametrize("exp, run_id", [("exp-1", "run-001"), ("exp-001", "run-1"), ("exp-001", "smoke-001"),
                                         ("scratch", "run-001"), ("experiment", "run-001")])
def test_bad_names_are_rejected(exp, run_id):
    with pytest.raises(ValueError):
        paths.check_names(exp, run_id)


def test_data_and_figure_folders(tmp_path):
    assert paths.data_dir(tmp_path) == tmp_path / "data"
    assert paths.figure_dir("02_pipeline", tmp_path) == tmp_path / "scratch" / "figures" / "02_pipeline"
