# colab-mnist-test

A small MNIST project that exercises the Jeeves Colab research workflow: write code here, check it in notebooks,
run it on a rented Google Colab machine, copy the results back, and keep a record. The experiment is a plumbing
test, not a research question; `spec.toml` states what is being tested.

## The five-stage development pipeline

| Stage | What | Files | Where it runs | Gate before moving on |
|---|---|---|---|---|
| 1 | Helper scripts | `data.py`, `augment.py`, `models.py`, `models_extra.py`, `quicktrain.py`, `runlib.py` | This computer, CPU | `uv run --extra local pytest` passes |
| 2 | Small runs of each helper in a notebook | `notebooks/01_helpers.ipynb` | CPU first, then a Colab GPU | The pictures look right and the checks print PASS |
| 3 | The training loop, written once | `trainer.py` (shared), `callbacks.py` | Imported by the notebook and the script | The notebook calls `trainer.fit`; nothing is copied |
| 4 | A few epochs in the notebook, with logic checks | `notebooks/pipeline.ipynb`, `checks.py` | Colab GPU (VS Code or browser) | Restart and Run All, then every check prints PASS |
| 5 | Deploy from the script | `train.py` | Colab, started by the command line | 200-step smoke test, checkpoint and resume, parity with the notebook |

`tests/test_parity.py` shows that `train.run` and a direct call to `trainer.fit` agree, and that a run stopped and
resumed from a checkpoint gives the same result as an uninterrupted run.

## Files

| File | Purpose |
|---|---|
| `spec.toml` | The experiment specification: question, data, hyperparameters, budget, stopping rules. |
| `trainer.py`, `callbacks.py` | The shared training loop with exact resume, and the checkpoint and heartbeat callbacks. |
| `checks.py` | Stage 4 logic checks: initial loss, overfit one batch, no overlap, loss falls, no NaN, above chance. |
| `data.py`, `augment.py` | Non-overlapping splits, MNIST loading, and the transforms. |
| `models.py`, `models_extra.py` | A CNN, an MLP, and a wide MLP kept in its own script. |
| `quicktrain.py` | Short training and evaluation helpers. |
| `runlib.py` | Run folders, heartbeat files, checkpoint discovery, spec loading. |
| `train.py` | The command-line run that writes a run folder. |
| `tools/run_notebook_local.py` | Runs a notebook on a local kernel one cell at a time. |
| `runs/<run-id>/` | Run records copied back from the machine. Checkpoints are not committed. |

## Run it

On this computer (CPU, small settings):

```
uv run --extra local pytest
uv run --extra local python tools/run_notebook_local.py notebooks/pipeline.ipynb
uv run --extra local jupyter lab notebooks/pipeline.ipynb
```

On Colab, by hand as the skill will do it:

```
colab new -s mnist --gpu T4
colab exec -s mnist -f setup_folder.py        # creates /content/mnist
colab upload train.py /content/mnist/train.py   # and the other .py files and spec.toml
colab exec -s mnist -f notebooks/pipeline.ipynb --timeout 240
```

Then start `train.py` as a background process on the machine, watch `heartbeat.json`, and download the run folder.
In VS Code, open a notebook, choose Select Kernel, Colab, New Colab Server, GPU, T4. The notebook's first code cell
mounts Drive and finds the scripts in `G:\My Drive\colab-mnist-test` (copy changed scripts there first).
