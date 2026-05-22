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
`env/curriculum_soft_quality`, a Python-derived target-progress score that
applies a 50% penalty for elevator, aileron, and rudder saturation. The stricter
`env/curriculum_quality` metric is still logged by the environment, but the
softer sweep metric keeps Protein from overvaluing clean low-target runs while
still discounting saturated control policies. The Dogfight-local runner remains
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

For longer official Protein sweeps, remember that sweep-space bounds control
later suggestions. `--train.total-timesteps` sets only the default trial.
Dogfight intentionally narrows its local sweep space from the PufferLib default:
`policy.hidden_size <= 256`, `policy.num_layers <= 4`, `train.horizon >= 32`,
and `train.total_timesteps` in the `25M..100M` range centered at `50M`. It also
sets `sweep.early_stop_min_steps = 40M`, so short promotion probes are not
killed before they have a chance to show post-1.9 progress. This avoids
pathological large-model, low-horizon trials that can look hung while still
training.

## Sweep Analysis and Hyper Benchmark

After a sweep, rank local W&B runs by target progress and export the
hyperparameters that produced the best short-run promotions. Set
`--analysis-project` to the W&B project being analyzed, for example `df38` for
the accepted baseline profile:

```bash
python ocean/dogfight/sweep_hypers.py \
  --analyze-wandb wandb \
  --analysis-project df38 \
  --analysis-out /tmp/df38_analysis.csv
```

The accepted hyper-selection profile from `df38` is now the benchmark baseline.
It pins `policy.hidden-size=128`, `policy.num-layers=3`,
`train.horizon=256`, `train.learning-rate=0.024`, `train.ent-coef=0.001`,
and `train.clip-coef=0.30`, plus the remaining train/vector defaults needed to
keep future code-change benchmarks from drifting. The old pre-`df38` INI
defaults are preserved as the historical `pre_df38_ini_defaults` profile.

List the saved benchmark profiles:

```bash
python ocean/dogfight/sweep_hypers.py --list-benchmark-profiles
```

Dry-run the current baseline benchmark commands before launching:

```bash
python ocean/dogfight/sweep_hypers.py \
  --benchmark \
  --steps 50000000 \
  --dry-run \
  --log-dir /tmp/dogfight_hyper_benchmark
```

Run five seeds with the current accepted baseline profile. Point
`--wandb-project` at the active sweep project:

```bash
python ocean/dogfight/sweep_hypers.py \
  --benchmark \
  --steps 50000000 \
  --log-dir /tmp/dogfight_hyper_benchmark \
  --wandb-project df38 \
  --wandb-group df38-hyper-baseline
```

To rerun the historical old-vs-current hyper comparison, include the historical
profile as the candidate:

```bash
python ocean/dogfight/sweep_hypers.py \
  --benchmark \
  --benchmark-candidate-profile pre_df38_ini_defaults \
  --steps 50000000 \
  --log-dir /tmp/dogfight_hyper_history \
  --wandb-project df38 \
  --wandb-group df38-hyper-history
```

After a paired benchmark with a real candidate profile finishes, use the
comparison gate:

```bash
python ocean/dogfight/sweep_hypers.py \
  --compare-benchmark /tmp/dogfight_hyper_benchmark/summary.csv
```

That gate passes when the candidate group has no extra failed runs, matches or
beats baseline median and best `max_target`, and keeps median
`selection_score` within 90% of baseline or better.

Once the profile is accepted, use the fixed-profile change benchmark for
behavior changes. This pins the selected model/training profile plus the other
Dogfight train/vector defaults, so old-vs-new comparisons cannot accidentally
drift because an unrelated hyperparameter changed.

Before a behavior change:

```bash
python ocean/dogfight/sweep_hypers.py \
  --change-benchmark \
  --steps 50000000 \
  --log-dir /tmp/dogfight_change_old \
  --wandb-project df39 \
  --wandb-group before-change
```

After the behavior change:

```bash
python ocean/dogfight/sweep_hypers.py \
  --change-benchmark \
  --steps 50000000 \
  --log-dir /tmp/dogfight_change_candidate \
  --wandb-project df39 \
  --wandb-group after-change
```

Compare the two fixed-profile summaries:

```bash
python ocean/dogfight/sweep_hypers.py \
  --compare-change-benchmark \
  /tmp/dogfight_change_old/summary.csv \
  /tmp/dogfight_change_candidate/summary.csv
```

The change gate first verifies each paired seed has exactly the same override
set in both summaries. It then applies the same no-extra-failures, median/best
`max_target`, and median `selection_score` checks to the candidate runs.

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
