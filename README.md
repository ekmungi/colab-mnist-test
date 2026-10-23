# colab-mnist-test

A small MNIST project that exercises the Jeeves Colab research workflow: write code here, run it on a rented
Google Colab machine through the Colab command-line tool, copy the results back, and keep a record.

The experiment is a plumbing test, not a research question. `spec.toml` states what is being tested.

## Files

| File | Purpose |
|---|---|
| `spec.toml` | The experiment specification: question, data, hyperparameters, budget, stopping rules. |
| `train.py` | Trains a small CNN with a heartbeat file, rolling checkpoints and exact resume. Runs on the Colab machine. |
| `runlib.py` | Helpers for run folders, heartbeat files and checkpoints. No torch, so it is tested locally. |
| `tests/` | Unit tests for `runlib.py`. Run with `uv run --with pytest pytest`. |
| `runs/<run-id>/` | Run records copied back from the machine: `metrics.json`, `heartbeat.json`, `train.log` and `record.md`. |

## Run on Colab (by hand, as the skill will)

```
colab new -s mnist --gpu T4
colab upload train.py /content/mnist/train.py
colab upload runlib.py /content/mnist/runlib.py
colab upload spec.toml /content/mnist/spec.toml
```

Then start `train.py` as a background process on the machine, watch `heartbeat.json`, and download the run folder.
Checkpoints stay on the machine (or in Drive) and are not committed.
