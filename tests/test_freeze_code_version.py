"""Tests for tools/freeze_code_version.py: the version string and the file it writes."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import freeze_code_version  # noqa: E402


def git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def make_repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    (tmp_path / "a.py").write_text("x")
    git(tmp_path, "add", "a.py")
    git(tmp_path, "commit", "-q", "-m", "first")


def test_a_clean_repository_gives_the_bare_commit_hash(tmp_path):
    make_repo(tmp_path)
    version = freeze_code_version.code_version(tmp_path)
    assert len(version) == 40 and "dirty" not in version


def test_uncommitted_changes_to_tracked_files_add_dirty(tmp_path):
    make_repo(tmp_path)
    (tmp_path / "a.py").write_text("changed")
    assert freeze_code_version.code_version(tmp_path).endswith(" dirty")


def test_freeze_writes_the_version_file_and_returns_the_same_string(tmp_path):
    make_repo(tmp_path)
    version = freeze_code_version.freeze(tmp_path)
    assert (tmp_path / "code_version.txt").read_text().strip() == version


def test_outside_a_repository_the_version_is_unknown(tmp_path):
    assert freeze_code_version.code_version(tmp_path) == "unknown"
