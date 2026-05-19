# Dogfight Hyperparameter Sweep

Use this when Dogfight tests are green but curriculum learning plateaus after
the early target. The goal is to decide whether PufferLib 4 training
hyperparameters are the bottleneck before changing Dogfight behavior again.

## Preflight

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
CC=clang bash ocean/dogfight/tests/run_all.sh
./build.sh dogfight
```

The local `.venv/bin/activate` hook sources `scripts/puffer_cuda_env.sh`, so
`source .venv/bin/activate && ./build.sh dogfight` is expected to work on WSL
g240. If `.venv` is recreated, run
`scripts/install_puffer_cuda_venv_hook.sh` again or source
`scripts/puffer_cuda_env.sh` before building.

GPU train and sweep runs must run outside the Codex sandbox so CUDA device
visibility, `ccache`, and NVIDIA libraries are available.

## Smoke Probe

Run one short baseline to verify GPU, logs, and parser:

```bash
python ocean/dogfight/sweep_hypers.py --steps 1000000 --trials baseline --log-dir /tmp/dogfight_sweep_smoke
cat /tmp/dogfight_sweep_smoke/summary.csv
```

## Overnight Probe

The fixed trial list is intentionally small and focused on optimizer stability:
learning rate, entropy, clip coefficient, value loss weight, replay ratio, two
combined stability settings, and one wider policy run.

```bash
python ocean/dogfight/sweep_hypers.py --steps 50000000 --trials all --log-dir /tmp/dogfight_sweep_50m
cat /tmp/dogfight_sweep_50m/summary.csv
```

For the first real hyperparameter search, cap each trial at `200M` timesteps and
log to W&B project `df37`. This does not try to finish training; it checks
whether Dogfight can move beyond the current post-promotion plateau under sane
PufferLib 4 optimizer settings.

```bash
python ocean/dogfight/sweep_hypers.py \
  --search random \
  --max-runs 1000 \
  --steps 200000000 \
  --wandb-project df37 \
  --wandb-group dogfight-200m-random \
  --log-dir /tmp/dogfight_sweep_200m_random

cat /tmp/dogfight_sweep_200m_random/summary.csv
```

The Dogfight config sets the official PufferLib sweep objective to
`env/curriculum_target`, so `pufferlib.pufferl sweep dogfight` can be used when
we want Protein to suggest hypers directly. The Dogfight-local runner remains
useful when we want resumable per-trial stdout logs and a CSV summary of
curriculum-specific diagnostics such as `base_stage_kills`, `player_ground`,
action saturation, and signed bias.

On WSL, official sweep smoke commands must keep Protein GP tensors on CPU while
the trainer uses GPU:

```bash
python -m pufferlib.pufferl sweep dogfight \
  --sweep.max-runs 1 \
  --sweep.gpus 1 \
  --train.gpus 1 \
  --train.total-timesteps 262144 \
  --sweep.use-gpu ""
```

Crash handling:

- each trial writes `trial.json` before launching training;
- each trial streams the full process output to `train.log`;
- after every completed or crashed trial, the runner rewrites `summary.csv` and
  `summary.jsonl`;
- if the runner is restarted with the same `--log-dir`, existing `train.log`
  files are parsed and skipped instead of overwritten;
- logs containing `traceback`, `exception`, `assert`, `nan`, or `crash` are
  marked failed in the summary.

For a longer confirmation of only the best candidates, pass a comma-separated
trial list:

```bash
python ocean/dogfight/sweep_hypers.py --steps 200000000 --skip-build \
  --trials baseline,stable_combo_a,stable_combo_b \
  --log-dir /tmp/dogfight_sweep_200m
```

If a run is interrupted, keep the trial directories and summarize what finished:

```bash
python ocean/dogfight/sweep_hypers.py --summarize /tmp/dogfight_sweep_50m
```

## Interpreting Results

A hyperparameter candidate is interesting if it:

- promotes past `0.90 -> 1.90`;
- improves post-promotion `final_base_stage_kills` or `best_base_stage_kills`
  over the current `~0.17` 200M plateau;
- keeps `final_player_ground <= 0.03`;
- reduces `final_avg_signed_bias` and action saturation rather than merely
  extending episodes.

If all focused trials plateau near the same post-promotion kill rate, treat the
blocker as Dogfight reward/curriculum/control behavior rather than PPO tuning.
