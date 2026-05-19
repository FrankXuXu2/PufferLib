# Dogfight Hyperparameter Sweep

Use this when Dogfight tests are green but curriculum learning plateaus after
the early target. The goal is to decide whether PufferLib 4 training
hyperparameters are the bottleneck before changing Dogfight behavior again.

## Preflight

```bash
cd /home/claude/PufferLib
CC=clang bash ocean/dogfight/tests/run_all.sh
PATH=.venv/bin:/usr/local/cuda-12.8/bin:$PATH CUDA_HOME=/usr/local/cuda-12.8 CCACHE_DIR=/tmp/ccache CC=clang ./build.sh dogfight
```

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
