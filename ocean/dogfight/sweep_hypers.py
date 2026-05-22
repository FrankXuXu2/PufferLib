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
from statistics import median

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = Path(os.environ.get("DOGFIGHT_SWEEP_DIR", "/tmp/dogfight_sweeps"))
SURFACE_SATURATION_KEYS = (
    "final_action_sat_elevator",
    "final_action_sat_aileron",
    "final_action_sat_rudder",
)
SURFACE_SATURATION_REJECT_THRESHOLD = 0.75
DEFAULT_BENCHMARK_SEEDS = (42, 43, 44, 45, 46)
OLD_INI_BENCHMARK_HYPERS: dict[str, str] = {
    "vec.total-agents": "4096",
    "vec.num-buffers": "8",
    "vec.num-threads": "8",
    "policy.hidden-size": "128",
    "policy.num-layers": "2",
    "train.minibatch-size": "65536",
    "train.horizon": "128",
    "train.learning-rate": "0.01788370841005079",
    "train.gamma": "0.9951941456096952",
    "train.gae-lambda": "0.8950950897628589",
    "train.ent-coef": "0.1604920157441345",
    "train.anneal-ent-coef": "0",
    "train.min-ent-coef-ratio": "0.1",
    "train.clip-coef": "0.41000268739635415",
    "train.vf-coef": "5.0",
    "train.vf-clip-coef": "3.484823007218511",
    "train.max-grad-norm": "3.8432803314849577",
    "train.anneal-lr": "1",
    "train.min-lr-ratio": "0",
    "train.beta1": "0.9848986194729068",
    "train.beta2": "0.998673462312489",
    "train.eps": "0.000002253309741136361",
    "train.prio-alpha": "0.5506490443590013",
    "train.prio-beta0": "1.0",
    "train.vtrace-rho-clip": "3.243009815021134",
    "train.vtrace-c-clip": "2.15402866995608",
    "train.replay-ratio": "2.8448025380102187",
}
SANE_BENCHMARK_HYPERS: dict[str, str] = {
    "policy.hidden-size": "128",
    "policy.num-layers": "3",
    "train.horizon": "256",
    "train.learning-rate": "0.024",
    "train.ent-coef": "0.001",
    "train.clip-coef": "0.30",
}
FIXED_BENCHMARK_HYPERS: dict[str, str] = {
    "vec.total-agents": "4096",
    "vec.num-buffers": "8",
    "vec.num-threads": "8",
    "policy.hidden-size": "128",
    "policy.num-layers": "3",
    "train.minibatch-size": "65536",
    "train.horizon": "256",
    "train.learning-rate": "0.024",
    "train.gamma": "0.9951941456096952",
    "train.gae-lambda": "0.8950950897628589",
    "train.ent-coef": "0.001",
    "train.anneal-ent-coef": "0",
    "train.min-ent-coef-ratio": "0.1",
    "train.clip-coef": "0.30",
    "train.vf-coef": "5.0",
    "train.vf-clip-coef": "3.484823007218511",
    "train.max-grad-norm": "3.8432803314849577",
    "train.anneal-lr": "1",
    "train.min-lr-ratio": "0",
    "train.beta1": "0.9848986194729068",
    "train.beta2": "0.998673462312489",
    "train.eps": "0.000002253309741136361",
    "train.prio-alpha": "0.5506490443590013",
    "train.prio-beta0": "1.0",
    "train.vtrace-rho-clip": "3.243009815021134",
    "train.vtrace-c-clip": "2.15402866995608",
    "train.replay-ratio": "2.8448025380102187",
}
CURRENT_BENCHMARK_PROFILE = "df38_sane_50m"
BENCHMARK_PROFILE_HISTORY: dict[str, dict[str, str]] = {
    "pre_df38_ini_defaults": OLD_INI_BENCHMARK_HYPERS,
    CURRENT_BENCHMARK_PROFILE: FIXED_BENCHMARK_HYPERS,
}

TRIALS: list[tuple[str, dict[str, str]]] = [
    ("baseline", {}),
    (
        "apricot_control",
        {
            "train.learning-rate": "0.01788370841005079",
            "train.ent-coef": "0.1604920157441345",
            "train.anneal-ent-coef": "0",
            "train.min-ent-coef-ratio": "0.1",
            "train.clip-coef": "0.41000268739635415",
            "train.vf-coef": "5",
            "train.vf-clip-coef": "3.484823007218511",
            "train.max-grad-norm": "3.8432803314849577",
            "train.replay-ratio": "2.8448025380102187",
            "train.beta1": "0.9848986194729068",
            "train.beta2": "0.998673462312489",
            "train.prio-alpha": "0.5506490443590013",
            "train.prio-beta0": "1",
            "train.vtrace-rho-clip": "3.243009815021134",
            "train.vtrace-c-clip": "2.15402866995608",
        },
    ),
    ("lr_00025", {"train.learning-rate": "0.00025"}),
    ("lr_00015", {"train.learning-rate": "0.00015"}),
    ("ent_0005", {"train.ent-coef": "0.005"}),
    ("ent_0008", {"train.ent-coef": "0.008"}),
    (
        "ent_anneal_0005",
        {
            "train.ent-coef": "0.005",
            "train.anneal-ent-coef": "1",
            "train.min-ent-coef-ratio": "0.05",
        },
    ),
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
            "train.anneal-ent-coef": "1",
            "train.min-ent-coef-ratio": "0.05",
            "train.clip-coef": "0.15",
            "train.vf-coef": "1.5",
        },
    ),
    (
        "stable_combo_b",
        {
            "train.learning-rate": "0.00015",
            "train.ent-coef": "0.008",
            "train.anneal-ent-coef": "1",
            "train.min-ent-coef-ratio": "0.10",
            "train.clip-coef": "0.20",
            "train.vf-coef": "1.0",
            "train.replay-ratio": "0.5",
        },
    ),
    ("policy_256", {"policy.hidden-size": "256"}),
]

RANDOM_SPACE: dict[str, tuple[str, float | int, float | int]] = {
    "train.learning-rate": ("log", 0.00008, 0.0008),
    "train.ent-coef": ("log", 0.0002, 0.04),
    "train.min-ent-coef-ratio": ("log", 0.01, 0.5),
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
    "train.anneal-ent-coef": ("1",),
    "train.replay-ratio": ("0.5", "1.0", "1.5", "2.0"),
    "policy.hidden-size": ("128", "256"),
    "policy.num-layers": ("2", "3"),
}

METRICS = [
    "SPS",
    "entropy",
    "curriculum_target",
    "base_stage_kills",
    "base_stage_eps",
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
TUI_PREFIXES = ("╭", "╰", "├", "┤", "│")


def maybe_float(value: object) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_selection_fields(summary: dict[str, object]) -> None:
    surface_values = [maybe_float(summary.get(key)) for key in SURFACE_SATURATION_KEYS]
    if any(value is None for value in surface_values):
        summary["surface_saturation"] = ""
        summary["selection_score"] = ""
        summary["rejected"] = True
        return

    surface_saturation = sum(float(value) for value in surface_values) / len(surface_values)
    best_kill_rate = maybe_float(summary.get("best_base_stage_kill_rate")) or 0.0
    max_target = maybe_float(summary.get("max_target")) or 0.9
    summary["surface_saturation"] = surface_saturation
    summary["selection_score"] = max_target + 0.25 * best_kill_rate - 2.0 * surface_saturation
    summary["rejected"] = surface_saturation > SURFACE_SATURATION_REJECT_THRESHOLD


def error_scan_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith(TUI_PREFIXES):
            continue
        lines.append(line)
    return "\n".join(lines).lower()


def add_derived_kill_rate(summary: dict[str, object], values: dict[str, list[float]]) -> None:
    if summary.get("final_base_stage_kill_rate") != "":
        return

    kills = values.get("base_stage_kills", [])
    eps = values.get("base_stage_eps", [])
    rates = [
        kill / episodes
        for kill, episodes in zip(kills, eps)
        if episodes > 0.0
    ]
    if not rates:
        return

    summary["final_base_stage_kill_rate"] = rates[-1]
    summary["best_base_stage_kill_rate"] = max(rates)


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

    lowered = error_scan_text(text)
    for marker in ("traceback", "exception", "assert", "nan", "crash"):
        if marker in lowered:
            summary["error"] = marker
            break

    values_by_metric: dict[str, list[float]] = {}
    for name, regex in METRIC_RES.items():
        values = [parse_number(m.group(1)) for m in regex.finditer(text)]
        values_by_metric[name] = values
        summary[f"final_{name}"] = values[-1] if values else ""
        summary[f"best_{name}"] = max(values) if values else ""

    add_derived_kill_rate(summary, values_by_metric)
    best_curriculum_target = maybe_float(summary.get("best_curriculum_target"))
    if best_curriculum_target is not None:
        summary["max_target"] = max(float(summary["max_target"]), best_curriculum_target)
    add_selection_fields(summary)
    return summary


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial",
        "group",
        "seed",
        "status",
        "seconds",
        "promotions",
        "max_target",
        "selection_score",
        "surface_saturation",
        "rejected",
        "last_promotion_kill_rate",
        "final_base_stage_kill_rate",
        "best_base_stage_kill_rate",
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
    jsonl_path = path.with_suffix(".jsonl")
    with jsonl_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


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
    if status == "ok" and summary.get("error"):
        status = f"failed:{summary['error']}"
    summary.update(
        {
            "trial": name,
            "status": status,
            "seconds": round(seconds, 1),
            "overrides": json.dumps(overrides, sort_keys=True),
        }
    )
    return summary


def summarize_existing_trial(name: str, trial_dir: Path) -> dict[str, object]:
    log_path = trial_dir / "train.log"
    row = parse_log(log_path)
    row["trial"] = name
    row["status"] = f"parsed_failed:{row['error']}" if row.get("error") else "parsed"
    row["seconds"] = ""
    row["overrides"] = ""
    trial_json = trial_dir / "trial.json"
    if trial_json.exists():
        meta = json.loads(trial_json.read_text())
        row["overrides"] = json.dumps(meta.get("overrides", {}), sort_keys=True)
    return row


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


def trials_from_summary(path: Path, top_k: int) -> list[tuple[str, dict[str, str]]]:
    rows: list[tuple[float, str, dict[str, str]]] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "")
            if not (status.startswith("ok") or status == "parsed"):
                continue
            if row.get("rejected") != "False":
                continue
            score = maybe_float(row.get("selection_score"))
            if score is None:
                continue
            overrides = json.loads(row.get("overrides") or "{}")
            rows.append((score, row["trial"], overrides))

    rows.sort(reverse=True, key=lambda item: item[0])
    return [(name, overrides) for _, name, overrides in rows[:top_k]]


def _unwrap_config_value(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def _scalar_config_value(config: dict[str, object], key: str) -> object:
    return _unwrap_config_value(config.get(key))


def _nested_config_value(config: dict[str, object], section: str, key: str) -> object:
    section_value = _unwrap_config_value(config.get(section, {}))
    if isinstance(section_value, dict) and key in section_value:
        return _unwrap_config_value(section_value[key])

    dotted = f"{section}.{key}"
    dashed = f"{section}.{key.replace('_', '-')}"
    for candidate in (dotted, dashed):
        if candidate in config:
            return _unwrap_config_value(config[candidate])
    return None


def _summary_metric(summary: dict[str, object], *keys: str) -> float | None:
    for key in keys:
        value = maybe_float(summary.get(key))
        if value is not None:
            return value
    return None


def collect_wandb_runs(wandb_dir: Path, project: str = "") -> list[dict[str, object]]:
    """Collect local W&B run summaries into rows useful for sweep analysis."""

    summary_paths = sorted(set(wandb_dir.glob("*run-*/files/wandb-summary.json")))
    rows: list[dict[str, object]] = []
    for summary_path in summary_paths:
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        target = _summary_metric(summary, "env/curriculum_target", "curriculum_target")
        if target is None:
            continue

        config_path = summary_path.with_name("config.yaml")
        try:
            config = yaml.safe_load(config_path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            config = {}
        if not isinstance(config, dict):
            config = {}

        if project and _scalar_config_value(config, "wandb_project") != project:
            continue

        surface_values = [
            _summary_metric(summary, f"env/action_sat_{surface}", f"action_sat_{surface}")
            for surface in ("elevator", "aileron", "rudder")
        ]
        surface_saturation: float | str
        if any(value is None for value in surface_values):
            surface_saturation = ""
        else:
            surface_saturation = sum(float(value) for value in surface_values) / len(surface_values)

        rows.append(
            {
                "run": summary_path.parents[1].name,
                "summary_path": str(summary_path),
                "target": target,
                "soft_quality": _summary_metric(summary, "env/curriculum_soft_quality") or 0.0,
                "strict_quality": _summary_metric(summary, "env/curriculum_quality") or 0.0,
                "agent_steps": _summary_metric(summary, "agent_steps", "global_step") or "",
                "sps": _summary_metric(summary, "SPS", "sps") or "",
                "base_stage_kills": _summary_metric(summary, "env/base_stage_kills", "base_stage_kills") or "",
                "surface_saturation": surface_saturation,
                "learning_rate": _nested_config_value(config, "train", "learning_rate"),
                "ent_coef": _nested_config_value(config, "train", "ent_coef"),
                "clip_coef": _nested_config_value(config, "train", "clip_coef"),
                "horizon": _nested_config_value(config, "train", "horizon"),
                "total_timesteps": _nested_config_value(config, "train", "total_timesteps"),
                "hidden_size": _nested_config_value(config, "policy", "hidden_size"),
                "num_layers": _nested_config_value(config, "policy", "num_layers"),
                "num_buffers": _nested_config_value(config, "vec", "num_buffers"),
            }
        )

    return rows


def rank_wandb_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    def sort_key(row: dict[str, object]) -> tuple[float, float, float, float]:
        return (
            maybe_float(row.get("target")) or 0.0,
            maybe_float(row.get("soft_quality")) or 0.0,
            maybe_float(row.get("strict_quality")) or 0.0,
            maybe_float(row.get("base_stage_kills")) or 0.0,
        )

    return sorted(rows, key=sort_key, reverse=True)


def write_wandb_analysis(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run",
        "target",
        "soft_quality",
        "strict_quality",
        "agent_steps",
        "sps",
        "base_stage_kills",
        "surface_saturation",
        "learning_rate",
        "ent_coef",
        "clip_coef",
        "horizon",
        "total_timesteps",
        "hidden_size",
        "num_layers",
        "num_buffers",
        "summary_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_seed_list(raw: str) -> list[int]:
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("--benchmark-seeds must include at least one seed")
    return seeds


def benchmark_profile_overrides(name: str) -> dict[str, str]:
    try:
        return dict(BENCHMARK_PROFILE_HISTORY[name])
    except KeyError:
        known = ", ".join(sorted(BENCHMARK_PROFILE_HISTORY))
        raise SystemExit(f"Unknown benchmark profile {name!r}. Known profiles: {known}") from None


def benchmark_trials(
    seeds: list[int] | tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS,
    baseline_profile: str = CURRENT_BENCHMARK_PROFILE,
    candidate_profile: str | None = None,
    candidate_overrides: dict[str, str] | None = None,
    sane_overrides: dict[str, str] | None = None,
) -> list[tuple[str, dict[str, str]]]:
    if candidate_profile and candidate_overrides:
        raise SystemExit("Use either candidate_profile or candidate_overrides, not both")
    if sane_overrides is not None:
        candidate_overrides = sane_overrides

    baseline = benchmark_profile_overrides(baseline_profile)
    trials: list[tuple[str, dict[str, str]]] = []
    for idx, seed in enumerate(seeds):
        overrides = {"train.seed": str(seed)}
        overrides.update(baseline)
        trials.append((f"baseline_{idx:03d}", overrides))

    if candidate_profile or candidate_overrides is not None:
        candidate = (
            dict(candidate_overrides)
            if candidate_overrides is not None
            else benchmark_profile_overrides(str(candidate_profile))
        )
        for idx, seed in enumerate(seeds):
            overrides = {"train.seed": str(seed)}
            overrides.update(candidate)
            trials.append((f"candidate_{idx:03d}", overrides))
    return trials


def change_benchmark_trials(
    seeds: list[int] | tuple[int, ...] = DEFAULT_BENCHMARK_SEEDS,
    fixed_overrides: dict[str, str] | None = None,
) -> list[tuple[str, dict[str, str]]]:
    fixed = fixed_overrides or benchmark_profile_overrides(CURRENT_BENCHMARK_PROFILE)
    trials: list[tuple[str, dict[str, str]]] = []
    for idx, seed in enumerate(seeds):
        overrides = {"train.seed": str(seed)}
        overrides.update(fixed)
        trials.append((f"change_{idx:03d}", overrides))
    return trials


def benchmark_metadata(trial_name: str, overrides: dict[str, str] | None = None) -> dict[str, str]:
    if "_" not in trial_name:
        return {"group": "", "seed": ""}
    group, _, suffix = trial_name.partition("_")
    if group not in {"baseline", "candidate", "sane", "change"}:
        return {"group": "", "seed": ""}
    seed = overrides.get("train.seed") if overrides else None
    return {"group": group, "seed": seed or suffix}


def _successful_status(status: object) -> bool:
    text = str(status)
    return text.startswith("ok") or text == "parsed"


def _benchmark_group(row: dict[str, object]) -> str:
    group = str(row.get("group") or "")
    if group:
        return group
    trial = str(row.get("trial") or "")
    if trial.startswith("baseline_"):
        return "baseline"
    if trial.startswith("candidate_"):
        return "candidate"
    if trial.startswith("sane_"):
        return "sane"
    return ""


def _numeric_values(rows: list[dict[str, object]], key: str) -> list[float]:
    return [value for value in (maybe_float(row.get(key)) for row in rows) if value is not None]


def _median_numeric(rows: list[dict[str, object]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    return median(values) if values else None


def _max_numeric(rows: list[dict[str, object]], key: str) -> float | None:
    values = _numeric_values(rows, key)
    return max(values) if values else None


def _row_overrides(row: dict[str, object]) -> dict[str, object]:
    raw = row.get("overrides") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _row_seed(row: dict[str, object]) -> str:
    overrides = _row_overrides(row)
    seed = row.get("seed") or overrides.get("train.seed")
    if seed is not None and str(seed) != "":
        return str(seed)

    trial = str(row.get("trial") or "")
    if "_" in trial:
        return trial.rsplit("_", 1)[-1]
    return ""


def _rows_by_seed(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    keyed: dict[str, dict[str, object]] = {}
    for row in rows:
        seed = _row_seed(row)
        if seed:
            keyed[seed] = row
    return keyed


def _benchmark_stats(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "runs": len(rows),
        "failed_runs": sum(not _successful_status(row.get("status")) for row in rows),
        "median_max_target": _median_numeric(rows, "max_target"),
        "best_max_target": _max_numeric(rows, "max_target"),
        "median_selection_score": _median_numeric(rows, "selection_score"),
    }


def compare_benchmark_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    comparison_group = "candidate" if any(_benchmark_group(row) == "candidate" for row in rows) else "sane"
    grouped = {"baseline": [], comparison_group: []}
    for row in rows:
        group = _benchmark_group(row)
        if group in grouped:
            grouped[group].append(row)

    failures: list[str] = []
    for group, group_rows in grouped.items():
        if not group_rows:
            failures.append(f"missing {group} benchmark rows")

    stats: dict[str, dict[str, object]] = {}
    for group, group_rows in grouped.items():
        stats[group] = _benchmark_stats(group_rows)

    baseline = stats["baseline"]
    candidate = stats[comparison_group]
    baseline_failed = int(baseline["failed_runs"])
    candidate_failed = int(candidate["failed_runs"])
    if candidate_failed > baseline_failed:
        failures.append(f"{comparison_group} failed runs {candidate_failed} > baseline failed runs {baseline_failed}")

    baseline_median = baseline["median_max_target"]
    candidate_median = candidate["median_max_target"]
    if (
        isinstance(baseline_median, float)
        and isinstance(candidate_median, float)
        and candidate_median < baseline_median
    ):
        failures.append(
            f"{comparison_group} median max_target {candidate_median} < baseline median max_target {baseline_median}"
        )

    baseline_best = baseline["best_max_target"]
    candidate_best = candidate["best_max_target"]
    if isinstance(baseline_best, float) and isinstance(candidate_best, float) and candidate_best < baseline_best:
        failures.append(f"{comparison_group} best max_target {candidate_best} < baseline best max_target {baseline_best}")

    baseline_score = baseline["median_selection_score"]
    candidate_score = candidate["median_selection_score"]
    if (
        isinstance(baseline_score, float)
        and isinstance(candidate_score, float)
        and candidate_score < 0.9 * baseline_score
    ):
        failures.append(
            f"{comparison_group} median selection_score {candidate_score} "
            f"< 90% of baseline median selection_score {baseline_score}"
        )

    return {"passed": not failures, "failures": failures, "stats": stats}


def compare_change_benchmark_rows(
    old_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    old_by_seed = _rows_by_seed(old_rows)
    candidate_by_seed = _rows_by_seed(candidate_rows)
    old_seeds = set(old_by_seed)
    candidate_seeds = set(candidate_by_seed)

    failures: list[str] = []
    missing_candidate = sorted(old_seeds - candidate_seeds)
    missing_old = sorted(candidate_seeds - old_seeds)
    if missing_candidate:
        failures.append(f"candidate missing seeds: {', '.join(missing_candidate)}")
    if missing_old:
        failures.append(f"old benchmark missing seeds: {', '.join(missing_old)}")

    common_seeds = sorted(old_seeds & candidate_seeds)
    if not common_seeds:
        failures.append("no paired seeds found between old and candidate summaries")
    old_common = [old_by_seed[seed] for seed in common_seeds]
    candidate_common = [candidate_by_seed[seed] for seed in common_seeds]
    for seed in common_seeds:
        old_overrides = _row_overrides(old_by_seed[seed])
        candidate_overrides = _row_overrides(candidate_by_seed[seed])
        if old_overrides != candidate_overrides:
            failures.append(f"overrides differ for seed {seed}")

    stats = {
        "old": _benchmark_stats(old_common),
        "candidate": _benchmark_stats(candidate_common),
    }
    old_stats = stats["old"]
    candidate_stats = stats["candidate"]

    old_failed = int(old_stats["failed_runs"])
    candidate_failed = int(candidate_stats["failed_runs"])
    if candidate_failed > old_failed:
        failures.append(f"candidate failed runs {candidate_failed} > old failed runs {old_failed}")

    old_median = old_stats["median_max_target"]
    candidate_median = candidate_stats["median_max_target"]
    if isinstance(old_median, float) and isinstance(candidate_median, float) and candidate_median < old_median:
        failures.append(f"candidate median max_target {candidate_median} < old median max_target {old_median}")

    old_best = old_stats["best_max_target"]
    candidate_best = candidate_stats["best_max_target"]
    if isinstance(old_best, float) and isinstance(candidate_best, float) and candidate_best < old_best:
        failures.append(f"candidate best max_target {candidate_best} < old best max_target {old_best}")

    old_score = old_stats["median_selection_score"]
    candidate_score = candidate_stats["median_selection_score"]
    if isinstance(old_score, float) and isinstance(candidate_score, float) and candidate_score < 0.9 * old_score:
        failures.append(
            f"candidate median selection_score {candidate_score} < 90% of old median selection_score {old_score}"
        )

    return {"passed": not failures, "failures": failures, "stats": stats}


def read_summary_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


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
    parser.add_argument("--from-summary", type=Path, help="Run top non-rejected rows from an existing summary CSV")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--benchmark", action="store_true", help="Run current benchmark profile trials")
    parser.add_argument("--change-benchmark", action="store_true", help="Run fixed-profile trials for old-vs-new code checks")
    parser.add_argument("--benchmark-seeds", default="42,43,44,45,46")
    parser.add_argument("--benchmark-baseline-profile", default=CURRENT_BENCHMARK_PROFILE)
    parser.add_argument("--benchmark-candidate-profile", default="")
    parser.add_argument("--list-benchmark-profiles", action="store_true")
    parser.add_argument("--compare-benchmark", type=Path, help="Exit 0 if sane benchmark matches or beats baseline")
    parser.add_argument(
        "--compare-change-benchmark",
        type=Path,
        nargs=2,
        metavar=("OLD_SUMMARY", "CANDIDATE_SUMMARY"),
        help="Exit 0 if candidate fixed-profile benchmark matches or beats old summary",
    )
    parser.add_argument("--analyze-wandb", type=Path, help="Rank local W&B runs from a wandb directory")
    parser.add_argument("--analysis-project", default="")
    parser.add_argument("--analysis-out", type=Path, help="CSV path for --analyze-wandb output")
    parser.add_argument("--analysis-top-k", type=int, default=25)
    args = parser.parse_args()

    if args.list_benchmark_profiles:
        for name in sorted(BENCHMARK_PROFILE_HISTORY):
            marker = " (current)" if name == CURRENT_BENCHMARK_PROFILE else ""
            print(f"{name}{marker}")
        return 0

    if args.compare_benchmark:
        result = compare_benchmark_rows(read_summary_rows(args.compare_benchmark))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.compare_change_benchmark:
        old_summary, candidate_summary = args.compare_change_benchmark
        result = compare_change_benchmark_rows(
            read_summary_rows(old_summary),
            read_summary_rows(candidate_summary),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["passed"] else 1

    if args.analyze_wandb:
        rows = rank_wandb_runs(collect_wandb_runs(args.analyze_wandb, project=args.analysis_project))
        if args.analysis_out:
            write_wandb_analysis(rows, args.analysis_out)
            print(f"Wrote {args.analysis_out}")
        else:
            for row in rows[: args.analysis_top_k]:
                print(
                    f"{row['run']} target={row['target']} soft={row['soft_quality']} "
                    f"steps={row['agent_steps']} hidden={row['hidden_size']} "
                    f"layers={row['num_layers']} horizon={row['horizon']} "
                    f"lr={row['learning_rate']} ent={row['ent_coef']} clip={row['clip_coef']}"
                )
        return 0

    if args.summarize:
        rows = []
        for log_path in sorted(args.summarize.glob("*/train.log")):
            row = summarize_existing_trial(log_path.parent.name, log_path.parent)
            rows.append(row)
        summary_path = args.summarize / "summary.csv"
        write_summary(rows, summary_path)
        print(f"Wrote {summary_path}")
        return 0

    benchmark_mode = bool(args.benchmark or args.change_benchmark)
    if args.benchmark and args.change_benchmark:
        raise SystemExit("--benchmark and --change-benchmark are mutually exclusive")
    if args.benchmark:
        trials = benchmark_trials(
            parse_seed_list(args.benchmark_seeds),
            baseline_profile=args.benchmark_baseline_profile,
            candidate_profile=args.benchmark_candidate_profile or None,
        )
    elif args.change_benchmark:
        trials = change_benchmark_trials(parse_seed_list(args.benchmark_seeds))
    elif args.from_summary:
        trials = trials_from_summary(args.from_summary, args.top_k)
    elif args.search == "random":
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
        trial_dir = args.log_dir / name
        if (trial_dir / "train.log").exists():
            row = summarize_existing_trial(name, trial_dir)
            print(f"Skipping existing trial {name}; parsed {trial_dir / 'train.log'}")
        else:
            row = run_trial(
                name,
                overrides,
                args.steps,
                args.log_dir,
                args.wandb_project or None,
                args.wandb_group or None,
            )
        if benchmark_mode:
            row.update(benchmark_metadata(name, overrides))
        rows.append(row)
        status = str(row["status"])
        failed = failed or not (status.startswith("ok") or status == "parsed")
        write_summary(rows, summary_path)
        print(f"Wrote {summary_path}", flush=True)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
