# colab-mnist-test

A small MNIST project that exercises the Jeeves Colab research workflow: write code here, check it in notebooks,
run it on a rented Google Colab machine, and keep a record of every run. The experiment is a plumbing test, not a
research question; `specs/exp-001_colab-plumbing.toml` states what is being tested.

## Where things live

This repository is `Dev/research/colab-mnist-test`. The Drive desktop app syncs the whole `Dev/research/` folder, so
the Colab machine reads the code from the mounted Drive and writes its results back into `artifacts/`.

| Folder or file | In Git | What it holds |
|---|---|---|
| Helper scripts at the root, `train.py` | yes | `data.py`, `augment.py`, `models.py`, `models_extra.py`, `trainer.py`, `callbacks.py`, `checks.py`, `quicktrain.py`, `runlib.py`, `paths.py` |
| `specs/` | yes | `exp-NNN_<slug>.toml`, one specification per experiment |
| `notebooks/` | yes | `01_helpers.ipynb` (stage 2), `02_pipeline.ipynb` (stages 3 and 4) |
| `tests/`, `tools/` | yes | unit tests; `run_notebook_local.py`, `freeze_code_version.py` |
| `README.md`, `docs/figures/` | yes | the method in words, and result pictures copied here by hand after review |
| `artifacts/` | **no** (git-ignored, synced to Drive) | everything else, see below |
| `code_version.txt` | no | commit hash written by `tools/freeze_code_version.py` before a Colab run |

```
artifacts/
  data/                         datasets and caches
  scratch/                      smoke-NNN runs, notebook-out/, figures/: throwaway, may be deleted
  exp-NNN/
    exp.md  findings.md         the question and conclusion; what was learned, for blog posts
    analysis/                   notebooks run on the results, tables and plots for the write-up
    run-NNN/
      config.json env.json git.txt     written once, never changed
      attempts.jsonl metrics_history.jsonl train.log    only appended to
      heartbeat.json                   overwritten (latest state)
      checkpoints/ figures/ predictions/ stdout.log
      metrics.json DONE record.md
```

`exp-NNN` and `run-NNN` numbers are never reused. A resume keeps its run id; a new attempt gets a new one.
Smoke tests live in `artifacts/scratch/smoke-NNN`. This repository is public, so nothing under `artifacts/` is ever
committed, and the repository holds no data, checkpoints or patient information.

## The five-stage development pipeline

| Stage | What | Files | Gate before moving on |
|---|---|---|---|
| 1 | Helper scripts | `data.py`, `augment.py`, `models.py`, `runlib.py`, `paths.py` | `pytest` passes |
| 2 | Small runs of each helper | `notebooks/01_helpers.ipynb` | The pictures look right and the checks print PASS |
| 3 | The training loop, written once | `trainer.py`, `callbacks.py` | The notebook calls `trainer.fit`; nothing is copied |
| 4 | A few epochs with logic checks | `notebooks/02_pipeline.ipynb`, `checks.py` | Restart and Run All, every check prints PASS |
| 5 | Deploy from the script | `train.py` | 200-step smoke test, checkpoint and resume, parity with the notebook |

`tests/test_parity.py` shows that `train.run` and a direct call to `trainer.fit` agree, and that a run stopped and
resumed from a checkpoint gives the same result as an uninterrupted run.

## Run it

Keep the Python environment outside the synced folder (it is large), and run on this computer (CPU, small settings):

```
set UV_PROJECT_ENVIRONMENT=C:\Users\ekmun\PythonEnvs\colab-mnist-test
uv run --extra local pytest
uv run --extra local python tools/run_notebook_local.py notebooks/02_pipeline.ipynb
```

On Colab, once Drive has synced the project: run `uv run python tools/freeze_code_version.py` on this computer, then
open `notebooks/02_pipeline.ipynb` in VS Code, choose Select Kernel, Colab, New Colab Server, GPU, T4, and run the
first code cell (it mounts Drive and finds the code). A script run, from a notebook cell or the Colab command line:

```
python train.py --spec specs/exp-001_colab-plumbing.toml --exp exp-001 --run-id run-003
python train.py --exp scratch --run-id smoke-001 --max-steps 200
```
