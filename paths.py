"""Where things live: one function per folder, so code, notebooks and tools never build paths by hand.

The layout follows the Jeeves Colab convention. The Drive desktop app syncs the whole Dev/research/ folder, so a
project's code and its git-ignored `artifacts/` folder (data/, scratch/, exp-NNN/run-NNN/) both reach Colab through
the mounted Drive, and anything Colab writes there comes back to this computer.
"""
import os
import re
from pathlib import Path

# Name of this project; the project folder is research/<PROJECT> on this computer and on Drive.
PROJECT = "colab-mnist-test"

# Where the mounted Drive appears on a Colab machine.
DRIVE_MOUNT = Path("/content/drive")

# Experiment folders are exp-NNN or the throwaway "scratch"; run folders are run-NNN or smoke-NNN.
EXP_PATTERN = re.compile(r"^(exp-\d{3}|scratch)$")
RUN_PATTERN = re.compile(r"^(run|smoke)-\d{3}$")


def project_root():
    """Returns the folder that holds this file: the project on this computer, the synced copy on Colab."""
    return Path(__file__).resolve().parent


def drive_project_dir(environ=None, mount=DRIVE_MOUNT):
    """Finds this project's synced folder on the mounted Drive, or returns None.

    Order: the DRIVE_PROJECT_DIR environment variable; MyDrive/research/<project>; and, for a computer folder
    synced by the Drive desktop app, Othercomputers/<computer name>/research/<project>.
    """
    environ = os.environ if environ is None else environ
    if environ.get("DRIVE_PROJECT_DIR"):
        return Path(environ["DRIVE_PROJECT_DIR"])
    candidates = [Path(mount) / "MyDrive" / "research" / PROJECT]
    candidates += sorted((Path(mount) / "Othercomputers").glob(f"*/research/{PROJECT}"))
    return next((c for c in candidates if c.is_dir()), None)


def artifact_root(environ=None, on_colab=None, mount=DRIVE_MOUNT):
    """Returns the artifacts folder for this machine.

    Order: the ARTIFACT_ROOT environment variable; on Colab the artifacts folder inside the synced Drive project
    when Drive is mounted, otherwise /content/artifacts (lost when the machine ends); on this computer
    <project>/artifacts. The arguments exist so tests can set the situation; leave them None in normal use.
    """
    environ = os.environ if environ is None else environ
    if environ.get("ARTIFACT_ROOT"):
        return Path(environ["ARTIFACT_ROOT"])
    on_colab = os.path.isdir("/content") if on_colab is None else on_colab
    if on_colab:
        synced = drive_project_dir(environ, mount)
        return synced / "artifacts" if synced else Path("/content/artifacts")
    return project_root() / "artifacts"


def check_names(exp, run_id):
    """Raises ValueError unless exp looks like exp-NNN (or scratch) and run_id like run-NNN (or smoke-NNN)."""
    if not EXP_PATTERN.match(exp):
        raise ValueError(f"experiment folder {exp!r} must be exp-NNN or scratch")
    if not RUN_PATTERN.match(run_id):
        raise ValueError(f"run id {run_id!r} must be run-NNN or smoke-NNN")
    if run_id.startswith("smoke") != (exp == "scratch"):
        raise ValueError("smoke runs live in scratch, and real runs live in an exp-NNN folder")


def exp_dir(exp, root=None):
    """Returns <root>/<exp>."""
    return Path(root or artifact_root()) / exp


def run_dir(exp, run_id, root=None):
    """Returns <root>/<exp>/<run_id> after checking the names."""
    check_names(exp, run_id)
    return exp_dir(exp, root) / run_id


def data_dir(root=None):
    """Returns <root>/data, where datasets and caches go."""
    return Path(root or artifact_root()) / "data"


def figure_dir(notebook, root=None):
    """Returns <root>/scratch/figures/<notebook>, where an exploration notebook saves its pictures."""
    return Path(root or artifact_root()) / "scratch" / "figures" / notebook
