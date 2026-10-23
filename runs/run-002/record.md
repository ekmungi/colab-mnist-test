# run-002

| Field | Value |
|---|---|
| Status | validated |
| Experiment | exp-001 (`spec.toml`) |
| Machine | free-tier T4, Python 3.13.16, torch 2.11.0+cu130 |
| Command | `python train.py --spec spec.toml --run-id run-002 --step-sleep 0.3`, killed hard at step about 340, then `--step-sleep 0.05` with the same run id |
| Failure | Process killed with SIGKILL after checkpoint 300. The heartbeat stopped at step 354 and kept `status: running`; its age grew from 19 s to 39 s. |
| Recovery | Relaunched with the same run id. Log line: `resumed from checkpoint-00000300.pt at step 300`. |
| Result | 1,290 steps, 3 epochs, test accuracy 0.9872, validation accuracy 0.9878, 96.8 s after the resume |
| Files here | `metrics.json`, `heartbeat.json`, `train.log`, `run-002.stdout` |

The resumed run finished and met the accuracy target. Its accuracy differs from run-001 by 0.0017, which is within what GPU nondeterminism allows, but exact reproduction after a resume was not tested.
