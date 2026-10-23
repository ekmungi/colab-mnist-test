# run-001

| Field | Value |
|---|---|
| Status | validated |
| Experiment | exp-001 (`spec.toml`) |
| Machine | free-tier T4, Python 3.13.16, torch 2.11.0+cu130 |
| Command | `python train.py --spec spec.toml --run-id run-001 --step-sleep 0.15`, started as a detached background process |
| Result | 1,290 steps, 3 epochs, test accuracy 0.9889, validation accuracy 0.9866, 245 s |
| Files here | `metrics.json`, `heartbeat.json`, `train.log`, `run-001.stdout`, `DONE` (the 14 MB checkpoint is not committed) |

An uninterrupted run, used as the reference for the resumed run-002.
