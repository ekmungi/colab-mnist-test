"""Small helpers for run folders, heartbeat files and checkpoint discovery, with no torch dependency so they can be tested anywhere."""
import json
import os
import re
import time
import tomllib
from pathlib import Path

# Matches checkpoint file names such as checkpoint-00000300.pt and captures the step number.
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)\.pt$")


def write_json_atomic(path, payload):
    """Writes payload as JSON to path through a temporary file, so a reader never sees a half-written file.

    Args:
        path: destination file path (str or Path).
        payload: any JSON-serialisable object.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_heartbeat(path, step, epoch, loss, last_checkpoint, status, now=None):
    """Writes the heartbeat file that tells an observer what the run is doing and how recently.

    Args:
        path: heartbeat file path.
        step: global training step reached.
        epoch: current epoch number, counting from 0.
        loss: latest training loss, or None before the first step.
        last_checkpoint: file name of the newest checkpoint, or None.
        status: "running", "done" or "failed".
        now: unix time to record; defaults to the current time (a parameter so tests can fix it).
    """
    write_json_atomic(path, {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "last_checkpoint": last_checkpoint,
        "status": status,
        "time": time.time() if now is None else now,
    })


def read_heartbeat(path):
    """Returns the heartbeat dict, or None when the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def heartbeat_age(heartbeat, now=None):
    """Returns seconds since the heartbeat was written, or None when there is no heartbeat."""
    if heartbeat is None:
        return None
    return (time.time() if now is None else now) - heartbeat["time"]


def checkpoint_path(run_dir, step):
    """Returns the checkpoint path for a step; the zero padding keeps names sorted by step."""
    return Path(run_dir) / f"checkpoint-{step:08d}.pt"


def list_checkpoints(run_dir):
    """Returns (step, path) pairs for every checkpoint in run_dir, oldest first."""
    found = []
    for entry in Path(run_dir).glob("checkpoint-*.pt"):
        match = CHECKPOINT_PATTERN.match(entry.name)
        if match:
            found.append((int(match.group(1)), entry))
    return sorted(found)


def latest_checkpoint(run_dir):
    """Returns the newest checkpoint path in run_dir, or None when there is none."""
    found = list_checkpoints(run_dir)
    return found[-1][1] if found else None


def prune_checkpoints(run_dir, keep):
    """Deletes all but the newest `keep` checkpoints and returns the removed paths.

    Args:
        run_dir: folder holding the checkpoints.
        keep: how many of the newest checkpoints to keep; must be at least 1.
    """
    if keep < 1:
        raise ValueError("keep must be at least 1")
    found = list_checkpoints(run_dir)
    removed = [path for _, path in found[:-keep]]
    for path in removed:
        path.unlink()
    return removed


def load_spec(path):
    """Reads an experiment specification written in TOML and returns it as a dict."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)
