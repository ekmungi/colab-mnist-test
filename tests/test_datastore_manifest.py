"""Tests for the source note format in datastore.py: one strict line format, and a note that verifies nothing is a problem."""
import hashlib

import pytest

import datastore

GOOD_DIGEST = hashlib.sha256(b"1").hexdigest()


def folder_with_note(tmp_path, note_lines, files=None):
    """Creates dataset folder `toy` holding {name: text} files and a source.md made of a header plus note_lines."""
    folder = tmp_path / "toy"
    folder.mkdir()
    for name, text in ({"one.txt": "1"} if files is None else files).items():
        (folder / name).write_text(text)
    (folder / "source.md").write_text("\n".join(["# toy", "", "- source: s", "", "## Files", "", *note_lines]) + "\n")
    return folder


def test_a_note_in_the_one_accepted_format_is_clean(tmp_path):
    assert datastore.check_manifest(folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"])) == []


@pytest.mark.parametrize("line", [
    f"- one.txt sha256={GOOD_DIGEST}",                 # one space
    f"- one.txt   sha256={GOOD_DIGEST}",                # three spaces
    f"- one.txt\tsha256={GOOD_DIGEST}",                # a tab
    f"- one.txt  sha256={GOOD_DIGEST.upper()}",        # uppercase hex
    f"- one.txt  sha256={GOOD_DIGEST} ",               # trailing space
    "- one.txt  sha256=abc123",                        # too short
    "- one.txt  sha256=",                              # empty
])
def test_a_malformed_file_line_is_reported_with_the_line_and_the_expected_format(tmp_path, line):
    problems = datastore.check_manifest(folder_with_note(tmp_path, [line, f"- one.txt  sha256={GOOD_DIGEST}"]))
    assert any("one.txt" in p and "sha256" in p and "two spaces" in p for p in problems)


def test_lines_of_the_note_that_are_not_file_lines_are_ignored(tmp_path):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}", "", "Some prose about the data."])
    assert datastore.check_manifest(folder) == []


def test_a_note_that_lists_no_files_is_a_problem(tmp_path):
    folder = folder_with_note(tmp_path, [], files={})
    (problem,) = datastore.check_manifest(folder)
    assert "no files" in problem and "source.md" in problem


def test_a_note_whose_only_file_line_is_malformed_is_a_problem_twice_over(tmp_path):
    problems = datastore.check_manifest(folder_with_note(tmp_path, ["- one.txt  sha256=abc"]))
    assert any("two spaces" in p for p in problems) and any("no files" in p for p in problems)


def test_a_dataset_file_the_note_does_not_list_is_a_problem(tmp_path):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"], {"one.txt": "1", "extra.txt": "2"})
    (problem,) = datastore.check_manifest(folder)
    assert "extra.txt" in problem and "not listed" in problem


def test_a_file_in_a_subfolder_that_the_note_does_not_list_is_named_with_its_path(tmp_path):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"])
    (folder / "sub").mkdir()
    (folder / "sub" / "deep.txt").write_text("d")
    (problem,) = datastore.check_manifest(folder)
    assert "sub/deep.txt" in problem


def test_the_markers_and_the_note_itself_do_not_count_as_unlisted_files(tmp_path):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"])
    (folder / datastore.READY_MARKER).write_text("unpacked\n")
    (folder / datastore.FETCHED_MARKER).write_text("fetched\n")
    assert datastore.check_manifest(folder) == []


def test_an_archive_must_be_listed_in_its_own_note(tmp_path):
    archive = tmp_path / "toy.zip"
    archive.write_bytes(b"PK")
    (tmp_path / "toy.source.md").write_text(f"# toy\n\n- other.zip  sha256={GOOD_DIGEST}\n")
    problems = datastore.check_manifest(archive)
    assert any("toy.zip" in p and "not listed" in p for p in problems)
    assert any("other.zip" in p and "missing" in p for p in problems)


def test_an_archive_note_that_lists_no_files_is_a_problem(tmp_path):
    archive = tmp_path / "toy.zip"
    archive.write_bytes(b"PK")
    (tmp_path / "toy.source.md").write_text("# toy\n")
    assert any("no files" in p for p in datastore.check_manifest(archive))


def test_the_notes_this_module_writes_pass_its_own_check(tmp_path):
    folder = tmp_path / "toy"
    (folder / "sub").mkdir(parents=True)
    (folder / "one.txt").write_text("1")
    (folder / "sub" / "two.txt").write_text("2")
    datastore.write_manifest(folder, source="s", licence="l", acquired_by="me")
    archive = tmp_path / "pack.zip"
    archive.write_bytes(b"PK")
    datastore.write_manifest(archive, source="s", licence="l", acquired_by="me")
    assert datastore.check_manifest(folder) == [] and datastore.check_manifest(archive) == []


@pytest.mark.parametrize("junk", ["desktop.ini", "Desktop.ini", ".DS_Store", "Thumbs.db", "thumbs.db", "sub/desktop.ini"])
def test_files_a_system_adds_to_a_synced_folder_do_not_count_as_unlisted_files(tmp_path, junk):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"])
    (folder / junk).parent.mkdir(parents=True, exist_ok=True)
    (folder / junk).write_text("system file\n")
    assert datastore.check_manifest(folder) == []


def test_an_ordinary_file_with_a_similar_name_is_still_unlisted(tmp_path):
    folder = folder_with_note(tmp_path, [f"- one.txt  sha256={GOOD_DIGEST}"])
    (folder / "desktop.ini.bak").write_text("data\n")
    assert len(datastore.check_manifest(folder)) == 1
