"""Trains a small CNN on MNIST and keeps a heartbeat file, rolling checkpoints and exact resume, so the Colab workflow can be tested end to end.

Runs on a Colab machine (torch and torchvision are preinstalled there). Everything it writes goes to
<out-dir>/<run-id>/: train.log, heartbeat.json, checkpoint-*.pt, metrics.json and a DONE marker.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import runlib

# MNIST channel statistics, the standard normalisation for this dataset.
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
TRAIN_SIZE, VAL_SIZE = 55000, 5000


class Net(nn.Module):
    """Two convolution layers and two linear layers; about 1.2 million parameters."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)  # 64 channels x 12 x 12 after the pooling step
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(torch.flatten(x, 1))))


def parse_args(argv=None):
    """Parses the command line; values from --spec's [train] table are used as defaults."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--spec", default=None)
    known, _ = pre.parse_known_args(argv)
    defaults = dict(epochs=3, batch_size=128, lr=1e-3, seed=0, checkpoint_every=100, heartbeat_every_s=5.0)
    if known.spec:
        defaults.update(runlib.load_spec(known.spec).get("train", {}))
    parser = argparse.ArgumentParser(parents=[pre])
    parser.add_argument("--run-id", default="run-001")
    parser.add_argument("--out-dir", default="/content/runs")
    parser.add_argument("--data-dir", default="/content/data")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = no limit), for smoke tests")
    parser.add_argument("--step-sleep", type=float, default=0.0, help="seconds to sleep per step, to make a short run last longer")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    for name, value in defaults.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    return parser.parse_args(argv)


def set_seeds(seed):
    """Seeds Python, NumPy and torch so a run with the same arguments repeats."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(run_dir, step, state, keep):
    """Writes a checkpoint atomically, then removes older ones beyond `keep`; returns the new path."""
    path = runlib.checkpoint_path(run_dir, step)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)
    runlib.prune_checkpoints(run_dir, keep)
    return path


def evaluate(model, loader, device):
    """Returns the accuracy of model over every batch in loader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.numel()
    model.train()
    return correct / total


def main(argv=None):
    """Runs the training, resuming from the newest checkpoint in the run folder when one exists."""
    args = parse_args(argv)
    run_dir = Path(args.out_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "train.log", "a", buffering=1)

    def say(message):
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        log.write(line + "\n")

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    full = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_set, val_set = random_split(full, [TRAIN_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=1000)
    test_loader = DataLoader(test_set, batch_size=1000)

    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    step = epoch = batch_in_epoch = 0
    last_checkpoint, loss_value = None, None

    resume_from = runlib.latest_checkpoint(run_dir)
    if resume_from is not None:
        # weights_only=True refuses to unpickle arbitrary objects, so a tampered checkpoint cannot run code.
        state = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, batch_in_epoch = state["step"], state["epoch"], state["batch_in_epoch"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        last_checkpoint = resume_from.name
        say(f"resumed from {resume_from.name} at step {step}")
    say(f"start device={device} torch={torch.__version__} epochs={args.epochs} batch={args.batch_size} seed={args.seed}")

    started, last_beat = time.time(), 0.0
    heartbeat_path = run_dir / "heartbeat.json"
    stopped_early = False
    while epoch < args.epochs and not stopped_early:
        order = torch.Generator().manual_seed(args.seed + epoch)  # same shuffle after a resume
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=order)
        for index, (x, y) in enumerate(loader):
            if index < batch_in_epoch:
                continue  # batches already trained before the checkpoint
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            step, batch_in_epoch, loss_value = step + 1, index + 1, loss.item()
            if args.step_sleep:
                time.sleep(args.step_sleep)
            if step % args.checkpoint_every == 0:
                last_checkpoint = save_checkpoint(run_dir, step, {
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "epoch": epoch, "batch_in_epoch": batch_in_epoch,
                    "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                }, args.keep_checkpoints).name
            if time.time() - last_beat >= args.heartbeat_every_s:
                runlib.write_heartbeat(heartbeat_path, step, epoch, loss_value, last_checkpoint, "running")
                last_beat = time.time()
            if args.max_steps and step >= args.max_steps:
                stopped_early = True
                break
        if stopped_early:
            break
        epoch, batch_in_epoch = epoch + 1, 0
        say(f"epoch {epoch} done step={step} loss={loss_value:.4f} val_acc={evaluate(model, val_loader, device):.4f}")

    metrics = {
        "run_id": args.run_id, "steps": step, "epochs_completed": epoch, "stopped_early": stopped_early,
        "val_accuracy": evaluate(model, val_loader, device), "test_accuracy": evaluate(model, test_loader, device),
        "seconds": round(time.time() - started, 1), "device": str(device), "torch": torch.__version__,
        "seed": args.seed, "resumed_from": resume_from.name if resume_from else None,
    }
    runlib.write_json_atomic(run_dir / "metrics.json", metrics)
    runlib.write_heartbeat(heartbeat_path, step, epoch, loss_value, last_checkpoint, "done")
    (run_dir / "DONE").write_text("finished\n")
    say(f"finished {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
