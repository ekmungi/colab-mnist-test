"""Writes code_version.txt (the commit hash, plus ' dirty' when tracked files have uncommitted changes) into the project folder.

The Drive desktop app syncs the project, so the Colab machine can read this file and every run records which code it used.
The file is git-ignored. Run it before starting a run on Colab.
"""
import subprocess
import sys
from pathlib import Path


def code_version(project_root):
    """Returns the commit hash, followed by ' dirty' when tracked files have uncommitted changes."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=project_root,
                           capture_output=True, text=True).stdout.strip()
    return f"{head} dirty" if dirty else head


def freeze(project_root):
    """Writes code_version.txt in project_root and returns the version string that was written."""
    version = code_version(project_root)
    (Path(project_root) / "code_version.txt").write_text(version + "\n", encoding="utf-8")
    return version


def main(argv=None):
    """Command line entry point: freeze_code_version.py (acts on the project that holds this tool)."""
    root = Path(__file__).resolve().parent.parent
    print(f"code_version.txt: {freeze(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
