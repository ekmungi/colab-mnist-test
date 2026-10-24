# colab-mnist-test

A small MNIST project that exercises the Jeeves Colab research workflow: write code here, check it in notebooks,
run it on a rented Google Colab machine, and keep a record of every run. The experiment is a plumbing test, not a
research question; `specs/exp-001_colab-plumbing.toml` states what is being tested.

## Where things live

One project has three places, and each place has one job.

| Place | Location on this computer | Synced with | Holds |
|---|---|---|---|
| Notes | `<vault>\01 PROJECTS\Colab MNIST Test\` | Obsidian | the plan, decisions, tasks and research notes |
| Code (this repository) | `C:\Users\<you>\Dev\colab-mnist-test\` | GitHub (public) | everything below except artifacts |
| Artifacts | `C:\Users\<you>\ResearchArtifacts\colab-mnist-test\` | Google Drive (two-way) | data, scratch work, experiments and their runs |

A Colab machine gets the code by cloning this public repository, and reads and writes artifacts through the mounted
Drive. Nothing from the artifacts folder is ever committed, so the repository holds no data, checkpoints or
patient information.

### This repository

| Folder or file | What it holds |
|---|---|
| Helper scripts at the root, `train.py` | `data.py`, `augment.py`, `models.py`, `models_extra.py`, `trainer.py`, `callbacks.py`, `checks.py`, `quicktrain.py`, `runlib.py`, `paths.py` |
| `specs/` | `exp-NNN_<slug>.toml`, one specification per experiment |
| `notebooks/` | `01_helpers.ipynb` (stage 2), `02_pipeline.ipynb` (stages 3 and 4) |
| `tests/`, `tools/` | unit tests; `run_notebook_local.py` runs a notebook cell by cell on this computer |
| `README.md`, `docs/figures/` | the method in words, and result pictures copied here by hand after review |

### The artifacts folder

```
ResearchArtifacts/colab-mnist-test/
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
Smoke tests live in `scratch/smoke-NNN`. `paths.py` is the only place that knows where the artifacts folder is:
`ARTIFACT_ROOT` if set, otherwise the Drive folder on Colab, otherwise `~/ResearchArtifacts/colab-mnist-test` here.

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

On this computer (CPU, small settings):

```
uv run --extra local pytest
uv run --extra local python tools/run_notebook_local.py notebooks/02_pipeline.ipynb
```

On Colab: push the code first, because the machine clones what is on GitHub. Then open
`notebooks/02_pipeline.ipynb` in VS Code, choose Select Kernel, Colab, New Colab Server, GPU, T4 (never TPU), and run
the first code cell. It clones this repository into `/content/colab-mnist-test` and mounts Drive for the artifacts;
approve the Drive pop-up within two minutes. A script run, from a notebook cell or the Colab command line:

```
python train.py --spec specs/exp-001_colab-plumbing.toml --exp exp-001 --run-id run-003
python train.py --exp scratch --run-id smoke-001 --max-steps 200
```
