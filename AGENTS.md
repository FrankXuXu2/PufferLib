# AGENTS.md

Guidance for autonomous agents working on the Dogfight port in this clone.

## Current Workspace

This machine is `g240`. The human is using it as an isolated work box so agents can work freely without risking the real PufferLib branch.

Writable working repo:

- `/home/claude/PufferLib`
- Remote: `origin git@github-frankxuxu2:FrankXuXu2/PufferLib.git`
- This is a clean PufferLib 4.0 clone and is the only repo where implementation work should land.

Reference repos:

- `/home/claude/dogfight3`
  - PufferLib 3.0-era Dogfight branch.
  - Trains well and has the useful self-play, league, reward, curriculum, and evaluation behavior.
  - Not mergeable as-is because PufferLib 4.0 changed the Ocean/env APIs.
- `/home/claude/dogfight4`
  - Prior PufferLib 4.0 Dogfight port attempt.
  - Structurally close to mergeable, with 4.0-style `ocean/dogfight` files and tests.
  - Training quality is poor, so do not assume its behavior is correct.
  - Its commit history is useful for understanding the attempted port.

Treat both reference repos as read-only. Copy/adapt code into `/home/claude/PufferLib` manually.

## Goal

Port the Dogfight environment into clean PufferLib 4.0 in a way that is mergeable upstream and trains well.

Use:

- `dogfight4` as the structural/API reference for PufferLib 4.0 layout.
- `dogfight3` as the behavioral reference for training quality, self-play, league, evaluation, curriculum, observations, rewards, and opponent handling.

The target is not a mechanical merge. The target is a careful 4.0-native port that preserves the parts of Dogfight 3.0 that made learning work.

## Safety Rules

- Do implementation work only in `/home/claude/PufferLib`.
- Keep implementation changes inside `ocean/dogfight/` whenever possible. The goal is an upstreamable Dogfight PR, and broad PufferLib core changes will make maintainers much more reluctant to merge it.
- Treat changes outside `ocean/dogfight/` as exceptional. Only make them when Dogfight cannot work correctly through existing 4.0 extension points, and then keep the patch minimal, tested, and clearly justified in the commit message/final note.
- Prefer Dogfight-local adapters, config, tests, and logging over repo-wide API changes. If a core change seems necessary, first look for a Dogfight-local workaround or an existing PufferLib 4.0 hook.
- Do not modify `/home/claude/dogfight3` or `/home/claude/dogfight4`.
- Do not add remotes or upstreams to official `PufferAI/PufferLib` or `Kinvert/PufferLib` in the working repo.
- Pushing is allowed only to the isolated fake-account fork: `FrankXuXu2/PufferLib`.
- Future agents may commit and push small checkpoints on branch `dogfight-port` without asking, as long as the remote still points only at `FrankXuXu2/PufferLib`.
- Before any push, verify:

```bash
git -C /home/claude/PufferLib remote -v
ssh -T github-frankxuxu2
```

Expected remote shape:

```text
origin git@github-frankxuxu2:FrankXuXu2/PufferLib.git (fetch)
origin git@github-frankxuxu2:FrankXuXu2/PufferLib.git (push)
```

This workspace is intentionally disposable, but avoid destructive Git operations anyway. Do not run `git reset --hard`, mass deletes, or history rewrites unless the human explicitly asks.

## Porting Strategy

Work in small, testable increments.

1. First get a minimal Dogfight 4.0 env present in `/home/claude/PufferLib/ocean/dogfight`.
2. Make `./build.sh dogfight` pass.
3. Make Dogfight regression tests pass.
4. Smoke-test normal 4.0 training without self-play.
5. Then port higher-level behavior from `dogfight3` feature by feature:
   - observation schemes
   - reward shaping
   - curriculum/stage logic
   - opponent observations
   - AutoAce/autopilot behavior
   - self-play pool behavior
   - league/eval tools

Avoid `git cherry-pick` from the reference repos. Read the code and history, identify a discrete behavior, then adapt it manually to 4.0 conventions.

Do not start the next phase until the current phase is green. If a change touches constructor/init arguments, observation sizes, action layout, reward fields, logging, or config keys, stop and think through the tests and call sites before editing.

## First Port Commit

Start by copying the structurally 4.0-compatible Dogfight files from `dogfight4`, not from `dogfight3`.

Initial copy set:

```text
/home/claude/dogfight4/config/dogfight.ini                 -> config/dogfight.ini
/home/claude/dogfight4/ocean/dogfight/autoace.h            -> ocean/dogfight/autoace.h
/home/claude/dogfight4/ocean/dogfight/autopilot.h          -> ocean/dogfight/autopilot.h
/home/claude/dogfight4/ocean/dogfight/binding.c            -> ocean/dogfight/binding.c
/home/claude/dogfight4/ocean/dogfight/dogfight.c           -> ocean/dogfight/dogfight.c
/home/claude/dogfight4/ocean/dogfight/dogfight.h           -> ocean/dogfight/dogfight.h
/home/claude/dogfight4/ocean/dogfight/dogfight_log.py      -> ocean/dogfight/dogfight_log.py
/home/claude/dogfight4/ocean/dogfight/dogfight_observations.h -> ocean/dogfight/dogfight_observations.h
/home/claude/dogfight4/ocean/dogfight/dogfight_render.h    -> ocean/dogfight/dogfight_render.h
/home/claude/dogfight4/ocean/dogfight/dogfight_spawn.h     -> ocean/dogfight/dogfight_spawn.h
/home/claude/dogfight4/ocean/dogfight/flightlib.h          -> ocean/dogfight/flightlib.h
/home/claude/dogfight4/ocean/dogfight/p40.glb              -> ocean/dogfight/p40.glb
/home/claude/dogfight4/ocean/dogfight/tests/               -> ocean/dogfight/tests/
```

Then run:

```bash
scripts/run_dogfight_tests.sh
```

Only after the 4.0 structural port builds/tests should agents start bringing behavior back from `dogfight3`:

```text
train_selfplay.py
train_dual_selfplay.py
league.py
league_manifest.py
elo_eval.py
arena.py
anchor_eval.py
reference_opponents/
```

Do not copy all 3.0 Python training code at once. Port one behavior/test at a time into 4.0-native patterns.

## TDD Loop

Use test-driven development for new behavior:

1. Write or port the smallest test that proves the behavior.
2. Run it and confirm it fails for the expected reason when practical.
3. Implement the minimum code needed to pass.
4. Re-run the narrow test.
5. Re-run the broader Dogfight suite before committing.

For physics/autopilot changes, prefer a deterministic C test under `ocean/dogfight/tests/` before tuning. For Python league/self-play behavior, port or write Python tests near the relevant logic before wiring it into training.

If a test is intentionally deferred, make that explicit in the test file or plan. Do not silently count skip stubs as real coverage.

## High-Signal Reference Paths

Dogfight 4.0 structural reference:

```text
/home/claude/dogfight4/ocean/dogfight/
/home/claude/dogfight4/config/dogfight.ini
/home/claude/dogfight4/ocean/dogfight/tests/run_all.sh
```

Dogfight 3.0 behavioral/self-play reference:

```text
/home/claude/dogfight3/pufferlib/ocean/dogfight/
/home/claude/dogfight3/pufferlib/ocean/dogfight/train_selfplay.py
/home/claude/dogfight3/pufferlib/ocean/dogfight/train_dual_selfplay.py
/home/claude/dogfight3/pufferlib/ocean/dogfight/league.py
/home/claude/dogfight3/pufferlib/ocean/dogfight/league_manifest.py
/home/claude/dogfight3/pufferlib/ocean/dogfight/elo_eval.py
/home/claude/dogfight3/pufferlib/ocean/dogfight/arena.py
```

PufferLib 4.0 self-play infrastructure already present:

```text
/home/claude/PufferLib/pufferlib/selfplay.py
```

## Useful Commands

Use these from the working repo unless noted:

```bash
cd /home/claude/PufferLib
```

Environment setup uses `uv`, not system `pip` or `python -m venv`:

```bash
uv venv --python 3.12 .venv
uv sync
uv pip install pytest
source .venv/bin/activate
python --version
```

Verified on this box: `.venv` uses Python 3.12.3, matching `pyproject.toml` (`requires-python = ">=3.10"`). If a Codex sandbox reports uv cache errors, run the command with filesystem/network approval; do not switch to system `pip`.

Inventory references:

```bash
scripts/dogfight_status.sh
find /home/claude/dogfight4/ocean/dogfight -maxdepth 3 -type f | sort
find /home/claude/dogfight3/pufferlib/ocean/dogfight -maxdepth 3 -type f | sort
git -C /home/claude/dogfight4 log --oneline -12
git -C /home/claude/dogfight3 log --oneline -12
```

Build:

```bash
./build.sh dogfight
```

Current 4.0 build syntax:

```bash
./build.sh ENV_NAME          # native CUDA trainer extension for one env
./build.sh ENV_NAME --cpu    # CPU fallback / torch-only path
./build.sh ENV_NAME --debug  # debug build
./build.sh ENV_NAME --local  # standalone executable with sanitizers
./build.sh ENV_NAME --fast   # optimized standalone executable
./build.sh all               # all envs, slow
```

`./build.sh dogfight` will fail until `ocean/dogfight/binding.c` and friends have been ported into the clean repo. That failure is expected before the first Dogfight implementation commit.

Use the `.venv` for Python commands. `build.sh` expects `python` on PATH, so activate first:

```bash
source .venv/bin/activate
```

On `g240`, `.venv/bin/activate` has a local hook that sources
`scripts/puffer_cuda_env.sh`. After activation, this should work without extra
inline environment variables:

```bash
source .venv/bin/activate
./build.sh dogfight
```

If `.venv` is recreated, reinstall that activation hook or source
`scripts/puffer_cuda_env.sh` before building:

```bash
scripts/install_puffer_cuda_venv_hook.sh
```

Current system-tool status on `g240`:

- `clang`, `libomp-dev`, and `ccache` are installed.
- `./build.sh cartpole --cpu` was verified with `CC=clang`.
- This session is WSL2. NVIDIA WSL GPU access works outside the Codex sandbox:
  - `nvidia-smi` sees `NVIDIA GeForce RTX 5060`.
  - PyTorch reports `torch.cuda.is_available() == True` and `device_count == 1`.
  - CUDA Toolkit 12.8 is installed from NVIDIA's WSL repo.
  - `nvcc --version` reports CUDA 12.8.
  - Native `./build.sh breakout` was verified.
  - A short native GPU breakout training smoke test reached ~3.2M SPS and showed GPU usage in the PufferLib TUI.
- In the Codex sandbox, GPU device visibility and `ccache` may fail unless commands run with escalated permissions. GPU train and sweep smokes must run outside the Codex sandbox.

Dogfight test suite once ported:

```bash
bash ocean/dogfight/tests/run_all.sh
```

For render tests, let the render run until the human exits it. Do not add artificial time limits or auto-close logic just to make the command finish.

After modifying Dogfight physics, observations, actions, rewards, or `dogfight.h`, run the C dogfight tests and any relevant Python flight tests. After modifying league/eval/self-play Python, run the focused Python tests for those modules once they exist in the clean repo.

General status:

```bash
git -C /home/claude/PufferLib status --short
git -C /home/claude/PufferLib diff --stat
```

Repository-level tests:

```bash
python -m pytest tests/test_import_performance.py
```

Do not assume all existing repo tests are healthy in this clean clone. Verified current failures:

- `tests/test_api.py` imports removed/stale `pufferlib.emulation`.
- `tests/test_sweep.py` has a pytest fixture mismatch.
- `tests/test_sweep_hyper.py` needs extra dependencies such as `pandas`.
- CUDA tests need the CUDA 12.8 environment variables shown below, and GPU commands may need to run outside the Codex sandbox.

Run broad tests only after auditing their current status. For Dogfight-only C/env changes, prefer Dogfight build + Dogfight tests first.

## Command Cookbook

These commands are the baseline for future autonomous agents. Adjust `dogfight` only if the config/env name changes.

Build Dogfight:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
./build.sh dogfight
```

Expected current result before Dogfight is ported:

```text
Error: environment 'dogfight' not found
```

CPU-build sanity check for an existing env:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
CC=clang ./build.sh cartpole --cpu
```

GPU-build sanity check for an existing env:

```bash
scripts/verify_gpu_breakout.sh
```

Run Dogfight tests/build once ported:

```bash
scripts/run_dogfight_tests.sh
```

Train Dogfight to a log once ported:

```bash
scripts/train_dogfight_log.sh 300000
```

Read the latest Dogfight log:

```bash
scripts/read_dogfight_logs.sh
```

Expected verified GPU behavior: breakout trains on the RTX 5060 and the TUI shows GPU usage. Last smoke test reached 262K steps at ~3.2M SPS.

Run Dogfight regression tests:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
bash ocean/dogfight/tests/run_all.sh
```

Expected current result before Dogfight is ported:

```text
bash: ocean/dogfight/tests/run_all.sh: No such file or directory
```

Smoke-test training in the foreground:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
python -m pufferlib.pufferl train dogfight
```

Expected current result before `config/dogfight.ini` is ported:

```text
ValueError: No config for env_name dogfight
```

Run training to a timestamped log:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
mkdir -p /tmp/dogfight_logs
python -m pufferlib.pufferl train dogfight 2>&1 | tee /tmp/dogfight_logs/train_$(date +%Y%m%d_%H%M%S).log
```

If native CUDA build is unavailable on `g240`, do not expect real training to work until `nvcc`/CUDA is installed or a working CPU/torch path is confirmed for Dogfight.

Run a bounded sweep. Do not run an unbounded sweep during agent work:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
python -m pufferlib.pufferl sweep dogfight \
  --sweep.max-runs 1 \
  --sweep.gpus 1 \
  --train.gpus 1 \
  --train.total-timesteps 262144 \
  --sweep.use-gpu "" \
  2>&1 | tee /tmp/dogfight_logs/sweep_$(date +%Y%m%d_%H%M%S).log
```

On WSL, keep `--sweep.use-gpu ""` in official sweep smoke commands so Protein
GP tensors remain on CPU while the trainer uses GPU.

Evaluate a checkpoint with the native eval path:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
python -m pufferlib.pufferl eval dogfight --load-model-path /path/to/model.bin
```

Run a head-to-head match if Dogfight is configured as a 2-agent/self-play env:

```bash
cd /home/claude/PufferLib
source .venv/bin/activate
python -m pufferlib.pufferl match dogfight \
  --load-model-path /path/to/policy_a.bin \
  --load-enemy-model-path /path/to/policy_b.bin \
  --num-games 4096
```

Read a training log:

```bash
tail -80 /tmp/dogfight_logs/train_*.log
strings /tmp/dogfight_logs/train_*.log | grep -iE 'error|exception|traceback|assert|shape|crash|nan' | tail -40
strings /tmp/dogfight_logs/train_*.log | grep -E 'SPS|score|perf|stage|clean|SELFPLAY|RATCHET|CHECKPOINT|ERROR' | tail -80
```

Watch a live log:

```bash
tail -f /tmp/dogfight_logs/train_YYYYMMDD_HHMMSS.log
```

Check Git safety before pushing:

```bash
git -C /home/claude/PufferLib remote -v
ssh -T github-frankxuxu2
git -C /home/claude/PufferLib status --short
```

## Known Dogfight Context

`dogfight3` trains well but cannot merge into 4.0 directly.

`dogfight4` is closer to mergeable but learns poorly. Its latest visible commits include:

```text
21281f62 Update Damping Coeffs
e0f74501 Add Smoothness Test
6736b053 Fix Quat Bugs - Add Flight Recovery Tests With Telemetry
e3b67ec5 More Dogfight4 Tests
771ed60f port dogfight_log.py and wire tests into run_all.sh
a92cf076 Port Python Tests to C
2f4b6243 Update Ini and Hard Code Obs Size For Now
40e1d15d Initial Dogfight 4.0
```

Important warning from prior work: older self-play had an opponent-observation bug where opponent observations used the wrong scheme/layout. When porting self-play or opponent policy inference, verify observation shape and layout for both player and opponent paths.

`dogfight4/ocean/dogfight/tests/test_self_play.py` is currently a skip stub because the 3.0 Python wrapper and checkpoint-driven opponent tests were not ported. Do not treat that as coverage.

Other known issues and traps from prior work:

- Curriculum logging can be swallowed by the TUI. Prefer structured logging or stderr for messages agents need to monitor during training.
- Verify g-force penalty behavior in training: bad tuning can make the agent dive toward targets instead of climbing/fighting correctly.
- There have been flight-control sign and authority questions around rudder, aileron, elevator, roll snap near vertical pitch, and steep-dive recovery oscillation. Do not “fix” these by intuition alone; write or run flight tests.
- Vertical spawn scenarios existed in earlier work but may not be wired into training. If porting them, verify both C-side spawn conditions and Python-side activation.
- `dogfight4` contains a skipped self-play test inventory. The real self-play validation still needs a 4.0-native wrapper/test strategy.

## Training And Evaluation Guidance

Standard training, once Dogfight is registered:

```bash
cd /home/claude/PufferLib
python -m pufferlib.pufferl train dogfight
```

Prefer config-file changes in `config/dogfight.ini` over long one-off CLI overrides when testing normal training behavior.

For self-play, the old 3.0 scripts in `dogfight3` are behavioral references, not drop-in 4.0 code. PufferLib 4.0 already has native self-play infrastructure in `pufferlib/selfplay.py`; adapt Dogfight to the 4.0 system unless there is a clear reason to port a local script.

When long training or sweep tasks are needed, always set a bounded exit strategy. Do not launch indefinite sweeps. Capture logs and inspect for:

```text
error
exception
traceback
assert
shape
crash
nan
```

Important log markers from the 3.0/4.0 Dogfight work:

```text
[EVAL] [MATCH] [RATING] [ANCHOR] [TRAIN] [VERIFY] [VERDICT]
[PROMOTE] [SELFPLAY] [RATCHET] [CHECKPOINT] [ROUND] [PHASE] [ERROR]
```

For quality comparison across sweeps, prefer fixed anchor evaluation over internal self-play strength whenever anchor eval has been ported. Internal strength can be non-comparable across runs.

## Agent Loop

When running autonomously:

1. Pick one small behavior or integration target.
2. Read the equivalent files in clean 4.0, `dogfight4`, and if behavioral, `dogfight3`.
3. Write or port a focused test first when behavior is changing.
4. Implement only the minimum needed in `/home/claude/PufferLib`.
5. Build and run the narrowest relevant tests.
6. Run the broader Dogfight test suite when available.
7. Commit to the fake-account fork with a clear message.
8. Push to `FrankXuXu2/PufferLib`.
9. Leave a short note in the final response describing what changed, what passed, and what remains risky.

Do not keep broad, uncommitted changes around for long. The point of this isolated account is to create reviewable checkpoints the human can inspect later.
