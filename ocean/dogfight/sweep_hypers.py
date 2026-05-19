#!/usr/bin/env python3
"""Run bounded Dogfight hyperparameter probes and summarize training logs.

This is intentionally simpler than the generic PufferLib sweep path. Dogfight
curriculum progress is easier to audit from full logs, and these fixed trials
answer one question: did the PufferLib 4 optimizer defaults move enough that the
current Dogfight ini is the bottleneck?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = Path(os.environ.get("DOGFIGHT_SWEEP_DIR", "/tmp/dogfight_sweeps"))

TRIALS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    ("lr_00025", {"train.learning-rate": "0.00025"}),
    ("lr_00015", {"train.learning-rate": "0.00015"}),
    ("ent_0005", {"train.ent-coef": "0.005"}),
    ("ent_0008", {"train.ent-coef": "0.008"}),
    ("clip_015", {"train.clip-coef": "0.15"}),
    ("clip_020", {"train.clip-coef": "0.20"}),
    ("vf_15", {"train.vf-coef": "1.5"}),
    ("vf_10", {"train.vf-coef": "1.0"}),
    ("replay_05", {"train.replay-ratio": "0.5"}),
    (
        "stable_combo_a",
        {
            "train.learning-rate": "0.00025",
            "train.ent-coef": "0.005",
            "train.clip-coef": "0.15",
            "train.vf-coef": "1.5",
        },
    ),
    (
        "stable_combo_b",
        {
            "train.learning-rate": "0.00015",
            "train.ent-coef": "0.008",
            "train.clip-coef": "0.20",
            "train.vf-coef": "1.0",
            "train.replay-ratio": "0.5",
        },
    ),
    ("policy_256", {"policy.hidden-size": "256"}),
]

RANDOM_SPACE: dict[str, tuple[str, float | int, float | int]] = {
    "train.learning-rate": ("log", 0.00008, 0.0008),
    "train.ent-coef": ("log", 0.0008, 0.02),
    "train.clip-coef": ("linear", 0.06, 0.30),
    "train.vf-coef": ("linear", 0.5, 4.0),
    "train.vf-clip-coef": ("linear", 0.2, 3.0),
    "train.max-grad-norm": ("linear", 0.5, 3.0),
    "train.beta1": ("linear", 0.90, 0.995),
    "train.beta2": ("linear", 0.90, 0.999),
    "train.vtrace-rho-clip": ("linear", 0.8, 4.0),
    "train.vtrace-c-clip": ("linear", 0.8, 4.0),
    "train.prio-alpha": ("linear", 0.5, 1.0),
    "train.prio-beta0": ("linear", 0.2, 1.0),
}

DISCRETE_RANDOM_SPACE: dict[str, tuple[str, ...]] = {
    "train.replay-ratio": ("0.5", "1.0", "1.5", "2.0"),
    "policy.hidden-size": ("128", "256"),
    "policy.num-layers": ("2", "3"),
}

METRICS = [
    "SPS",
    "entropy",
    "curriculum_target",
    "base_stage_kills",
    "base_stage_kill_rate",
    "episode_length",
    "player_ground",
    "opponent_ground",
    "avg_signed_bias",
    "avg_control_rate",
    "action_abs_elevator",
    "action_abs_aileron",
    "action_abs_rudder",
    "action_abs_trigger",
    "action_sat_elevator",
    "action_sat_aileron",
    "action_sat_rudder",
    "action_sat_trigger",
]

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
PROMOTION_RE = re.compile(
    r"\[CURRICULUM\].*?target\s+([0-9.]+)\s+->\s+([0-9.]+)"
    r"\s+kill_rate=([0-9.]+)\s+episodes=([0-9.]+)"
)


def parse_number(raw: str) -> float:
    value = raw.strip()
    multiplier = 1.0
    if value.endswith("K"):
        multiplier = 1_000.0
        value = value[:-1]
    elif value.endswith("M"):
        multiplier = 1_000_000.0
        value = value[:-1]
    elif value.endswith("B"):
        multiplier = 1_000_000_000.0
        value = value[:-1]
    return float(value) * multiplier


def metric_regex(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_/]){re.escape(name)}\s+(-?[0-9.]+[KMB]?)")


METRIC_RES = {name: metric_regex(name) for name in METRICS}


def parse_log(path: Path) -> dict[str, object]:
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))
    summary: dict[str, object] = {
        "log": str(path),
        "promotions": 0,
        "max_target": 0.9,
        "last_promotion_kill_rate": "",
        "last_promotion_episodes": "",
        "error": "",
    }

    for match in PROMOTION_RE.finditer(text):
        summary["promotions"] = int(summary["promotions"]) + 1
        summary["max_target"] = max(float(summary["max_target"]), float(match.group(2)))
        summary["last_promotion_kill_rate"] = match.group(3)
        summary["last_promotion_episodes"] = match.group(4)

    lowered = text.lower()
    for marker in ("traceback", "exception", "assert", "nan", "crash"):
        if marker in lowered:
            summary["error"] = marker
            break

    for name, regex in METRIC_RES.items():
        values = [parse_number(m.group(1)) for m in regex.finditer(text)]
        summary[f"final_{name}"] = values[-1] if values else ""
        summary[f"best_{name}"] = max(values) if values else ""

    return summary


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial",
        "status",
        "seconds",
        "promotions",
        "max_target",
        "last_promotion_kill_rate",
        "final_base_stage_kills",
        "best_base_stage_kills",
        "final_player_ground",
        "final_episode_length",
        "final_avg_signed_bias",
        "final_action_sat_elevator",
        "final_action_sat_aileron",
        "final_action_sat_rudder",
        "final_action_sat_trigger",
        "final_SPS",
        "error",
        "log",
        "overrides",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def command_for_trial(
    steps: int,
    overrides: dict[str, str],
    wandb_project: str | None = None,
    wandb_group: str | None = None,
) -> list[str]:
    cmd = [
        str(REPO / ".venv/bin/python"),
        "-m",
        "pufferlib.pufferl",
        "train",
        "dogfight",
        "--train.total-timesteps",
        str(steps),
    ]
    if wandb_project:
        cmd.extend(["--wandb", "--wandb-project", wandb_project])
    if wandb_group:
        cmd.extend(["--wandb-group", wandb_group])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def env_for_run() -> dict[str, str]:
    env = os.environ.copy()
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda-12.8")
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{REPO / '.venv/bin'}:{cuda_home}/bin:/usr/lib/wsl/lib:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"/usr/lib/wsl/lib:{cuda_home}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("CCACHE_DIR", "/tmp/ccache")
    env.setdefault("CC", "clang")
    return env


def run_trial(
    name: str,
    overrides: dict[str, str],
    steps: int,
    out_dir: Path,
    wandb_project: str | None,
    wandb_group: str | None,
) -> dict[str, object]:
    trial_dir = out_dir / name
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"
    meta_path = trial_dir / "trial.json"
    cmd = command_for_trial(steps, overrides, wandb_project, wandb_group)
    meta = {
        "trial": name,
        "steps": steps,
        "overrides": overrides,
        "command": cmd,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    start = time.time()
    status = "ok"
    with log_path.open("w") as log:
        print(f"\n== {name} ==", flush=True)
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env_for_run(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    seconds = time.time() - start
    if rc != 0:
        status = f"failed:{rc}"

    summary = parse_log(log_path)
    summary.update(
        {
            "trial": name,
            "status": status,
            "seconds": round(seconds, 1),
            "overrides": json.dumps(overrides, sort_keys=True),
        }
    )
    return summary


def selected_trials(names: str) -> list[tuple[str, dict[str, str]]]:
    if names == "all":
        return TRIALS
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    trial_map = dict(TRIALS)
    missing = [name for name in wanted if name not in trial_map]
    if missing:
        raise SystemExit(f"Unknown trial(s): {', '.join(missing)}")
    return [(name, trial_map[name]) for name in wanted]


def sample_value(kind: str, low: float | int, high: float | int, rng: random.Random) -> str:
    if kind == "log":
        value = 10 ** rng.uniform(math.log10(float(low)), math.log10(float(high)))
    elif kind == "linear":
        value = rng.uniform(float(low), float(high))
    else:
        raise ValueError(f"unknown sample kind {kind}")
    return f"{value:.6g}"


def random_trials(max_runs: int, seed: int) -> list[tuple[str, dict[str, str]]]:
    rng = random.Random(seed)
    trials: list[tuple[str, dict[str, str]]] = [("baseline", {})]
    for idx in range(1, max_runs):
        overrides = {
            key: sample_value(kind, low, high, rng)
            for key, (kind, low, high) in RANDOM_SPACE.items()
        }
        for key, choices in DISCRETE_RANDOM_SPACE.items():
            overrides[key] = rng.choice(choices)
        trials.append((f"random_{idx:04d}", overrides))
    return trials


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--trials", default="all", help="Comma list of trials, or all")
    parser.add_argument("--search", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", type=Path, help="Summarize an existing sweep directory")
    args = parser.parse_args()

    if args.summarize:
        rows = []
        for log_path in sorted(args.summarize.glob("*/train.log")):
            row = parse_log(log_path)
            trial_json = log_path.with_name("trial.json")
            row["trial"] = log_path.parent.name
            row["status"] = "parsed"
            row["seconds"] = ""
            row["overrides"] = ""
            if trial_json.exists():
                meta = json.loads(trial_json.read_text())
                row["overrides"] = json.dumps(meta.get("overrides", {}), sort_keys=True)
            rows.append(row)
        summary_path = args.summarize / "summary.csv"
        write_summary(rows, summary_path)
        print(f"Wrote {summary_path}")
        return 0

    if args.search == "random":
        max_runs = args.max_runs or 100
        trials = random_trials(max_runs, args.seed)
    else:
        trials = selected_trials(args.trials)
        if args.max_runs:
            trials = trials[: args.max_runs]
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for name, overrides in trials:
            print(name)
            print(
                "  "
                + " ".join(
                    shlex.quote(part)
                    for part in command_for_trial(
                        args.steps,
                        overrides,
                        args.wandb_project or None,
                        args.wandb_group or None,
                    )
                )
            )
        return 0

    if not args.skip_build:
        subprocess.check_call(["./build.sh", "dogfight"], cwd=REPO, env=env_for_run())

    rows: list[dict[str, object]] = []
    summary_path = args.log_dir / "summary.csv"
    failed = False
    for name, overrides in trials:
        row = run_trial(
            name,
            overrides,
            args.steps,
            args.log_dir,
            args.wandb_project or None,
            args.wandb_group or None,
        )
        rows.append(row)
        failed = failed or not str(row["status"]).startswith("ok")
        write_summary(rows, summary_path)
        print(f"Wrote {summary_path}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
