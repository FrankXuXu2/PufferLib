"""Tests for ocean/dogfight/sweep_hypers.py saturation-aware summaries."""

from __future__ import annotations

import math
import json
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "ocean/dogfight"))

import sweep_hypers


def _parse_text(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "train.log"
        path.write_text(text)
        return sweep_hypers.parse_log(path)


def test_surface_saturation_score_accepts_unsaturated_run(failures):
    row = _parse_text(
        """
        [CURRICULUM] target 1.90 -> 2.90 kill_rate=0.95 episodes=2048
        SPS 1.2M curriculum_target 2.90 base_stage_kill_rate 0.80 entropy 1.4
        action_sat_elevator 0.30 action_sat_aileron 0.45 action_sat_rudder 0.60
        action_sat_trigger 1.00
        """
    )

    expected_surface = (0.30 + 0.45 + 0.60) / 3.0
    expected_score = 2.90 + 0.25 * 0.80 - 2.0 * expected_surface

    if row["rejected"] is not False:
        failures.append(f"expected unsaturated row to be accepted, got {row['rejected']!r}")
    if not math.isclose(row["surface_saturation"], expected_surface, abs_tol=1e-9):
        failures.append(f"surface_saturation mismatch: {row['surface_saturation']!r}")
    if not math.isclose(row["selection_score"], expected_score, abs_tol=1e-9):
        failures.append(f"selection_score mismatch: {row['selection_score']!r}")


def test_surface_saturation_rejects_saturated_run(failures):
    row = _parse_text(
        """
        SPS 1.2M curriculum_target 1.90 base_stage_kill_rate 0.55
        action_sat_elevator 0.80 action_sat_aileron 0.90 action_sat_rudder 0.70
        action_sat_trigger 0.00
        """
    )

    if row["rejected"] is not True:
        failures.append(f"expected saturated row to be rejected, got {row['rejected']!r}")
    if not math.isclose(row["surface_saturation"], 0.80, abs_tol=1e-9):
        failures.append(f"surface_saturation mismatch: {row['surface_saturation']!r}")


def test_surface_saturation_rejects_missing_required_metrics(failures):
    row = _parse_text(
        """
        SPS 1.2M curriculum_target 1.90 base_stage_kill_rate 0.55
        action_sat_elevator 0.20 action_sat_aileron 0.30 action_sat_trigger 1.00
        """
    )

    if row["rejected"] is not True:
        failures.append("expected missing rudder saturation to reject the row")
    if row["surface_saturation"] != "":
        failures.append(f"missing metrics should leave surface_saturation blank, got {row['surface_saturation']!r}")
    if row["selection_score"] != "":
        failures.append(f"missing metrics should leave selection_score blank, got {row['selection_score']!r}")


def test_startup_dashboard_nan_is_not_an_error(failures):
    row = _parse_text(
        """
        ╭──────────────────────────────────────────────────────────────────────────────╮
        │  Env           dogfight      GPU        0ms   0%    policy              nan  │
        │  Params          102.4K      Env        0ms   0%    value               nan  │
        │  Steps              0.0    Train      526ms  99%    entropy             nan  │
        ╰──────────────────────────────────────────────────────────────────────────────╯
        curriculum_target 1.90 base_stage_kill_rate 0.80 entropy 1.4
        action_sat_elevator 0.30 action_sat_aileron 0.45 action_sat_rudder 0.60
        """
    )

    if row["error"] != "":
        failures.append(f"dashboard startup nan should not mark error, got {row['error']!r}")
    if row["rejected"] is not False:
        failures.append(f"valid non-saturated metrics should not reject, got {row['rejected']!r}")


def test_base_stage_kill_rate_derives_from_kills_and_eps(failures):
    row = _parse_text(
        """
        [CURRICULUM] epoch=4 target 0.90 -> 1.90 kill_rate=0.910 episodes=2754
        base_stage_kills 0.10 base_stage_eps 0.50
        action_sat_elevator 0.30 action_sat_aileron 0.45 action_sat_rudder 0.60
        base_stage_kills 0.40 base_stage_eps 0.50
        action_sat_elevator 0.20 action_sat_aileron 0.30 action_sat_rudder 0.40
        """
    )

    expected_surface = (0.20 + 0.30 + 0.40) / 3.0
    expected_score = 1.90 + 0.25 * 0.80 - 2.0 * expected_surface

    if not math.isclose(row["final_base_stage_kill_rate"], 0.80, abs_tol=1e-9):
        failures.append(f"final derived kill rate mismatch: {row['final_base_stage_kill_rate']!r}")
    if not math.isclose(row["best_base_stage_kill_rate"], 0.80, abs_tol=1e-9):
        failures.append(f"best derived kill rate mismatch: {row['best_base_stage_kill_rate']!r}")
    if not math.isclose(row["selection_score"], expected_score, abs_tol=1e-9):
        failures.append(f"derived kill rate not included in score: {row['selection_score']!r}")


def test_random_trials_include_entropy_annealing_space(failures):
    trials = dict(sweep_hypers.random_trials(max_runs=3, seed=37))

    apricot = dict(sweep_hypers.TRIALS)["apricot_control"]
    if apricot.get("train.ent-coef") != "0.1604920157441345":
        failures.append("apricot_control should pin the high-entropy reference")

    random_rows = [overrides for name, overrides in trials.items() if name.startswith("random_")]
    if len(random_rows) != 2:
        failures.append(f"expected two random trials, got {len(random_rows)}")
    for idx, overrides in enumerate(random_rows, start=1):
        if overrides.get("train.anneal-ent-coef") != "1":
            failures.append(f"random trial {idx} missing entropy annealing")
        if "train.min-ent-coef-ratio" not in overrides:
            failures.append(f"random trial {idx} missing min entropy ratio")
        ent_coef = float(overrides["train.ent-coef"])
        if not 0.0002 <= ent_coef <= 0.04:
            failures.append(f"random trial {idx} ent_coef out of low-entropy range: {ent_coef}")


def test_random_trial_overrides_parse_as_dogfight_cli(failures):
    import pufferlib.pufferl as pufferl

    overrides = dict(sweep_hypers.random_trials(max_runs=2, seed=37))["random_0001"]
    argv = ["pufferl.py"]
    for key, value in overrides.items():
        argv.extend([f"--{key}", value])

    old_argv = sys.argv
    try:
        sys.argv = argv
        try:
            args = pufferl.load_config("dogfight")
        except SystemExit as exc:
            failures.append(f"random override CLI parse failed with exit {exc.code}")
            return
    finally:
        sys.argv = old_argv

    if args["train"]["anneal_ent_coef"] != 1:
        failures.append("entropy annealing override did not parse")
    if not isinstance(args["train"]["vf_coef"], float):
        failures.append(f"vf_coef should parse as float, got {type(args['train']['vf_coef']).__name__}")
    if not isinstance(args["train"]["prio_beta0"], float):
        failures.append(f"prio_beta0 should parse as float, got {type(args['train']['prio_beta0']).__name__}")


def test_trials_from_summary_selects_top_non_rejected(failures):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "summary.csv"
        path.write_text(
            "trial,status,selection_score,rejected,overrides\n"
            'bad,parsed,9.0,True,"{""train.ent-coef"": ""0.1""}"\n'
            'low,failed:2,8.0,False,"{""train.ent-coef"": ""0.2""}"\n'
            'second,ok,1.5,False,"{""train.ent-coef"": ""0.3""}"\n'
            'first,parsed,2.0,False,"{""train.ent-coef"": ""0.4""}"\n'
        )

        trials = sweep_hypers.trials_from_summary(path, top_k=2)

    expected = [
        ("first", {"train.ent-coef": "0.4"}),
        ("second", {"train.ent-coef": "0.3"}),
    ]
    if trials != expected:
        failures.append(f"unexpected selected trials: {trials!r}")


def _write_wandb_run(root: Path, name: str, summary: dict, config: dict) -> Path:
    files = root / name / "files"
    files.mkdir(parents=True)
    (files / "wandb-summary.json").write_text(json.dumps(summary))
    (files / "config.yaml").write_text(json.dumps(config))
    return root / name


def test_wandb_analysis_ranks_high_targets_and_extracts_hypers(failures):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base_config = {
            "wandb_project": {"value": "df38"},
            "train": {
                "value": {
                    "learning_rate": 0.024,
                    "ent_coef": 0.001,
                    "clip_coef": 0.30,
                    "horizon": 256,
                    "total_timesteps": 50_000_000,
                },
            },
            "policy": {"value": {"hidden_size": 128, "num_layers": 3}},
            "vec": {"value": {"num_buffers": 6}},
        }
        _write_wandb_run(
            root,
            "run-low",
            {
                "env/curriculum_target": 5.9,
                "env/curriculum_soft_quality": 0.35,
                "env/curriculum_quality": 0.30,
                "env/base_stage_kills": 0.40,
                "env/action_sat_elevator": 0.2,
                "env/action_sat_aileron": 0.1,
                "env/action_sat_rudder": 0.3,
                "agent_steps": 25_000_000,
                "SPS": 1_200_000,
            },
            base_config,
        )
        _write_wandb_run(
            root,
            "run-high",
            {
                "env/curriculum_target": 8.9,
                "env/curriculum_soft_quality": 0.41,
                "env/curriculum_quality": 0.30,
                "env/base_stage_kills": 0.60,
                "env/action_sat_elevator": 0.4,
                "env/action_sat_aileron": 0.2,
                "env/action_sat_rudder": 0.3,
                "agent_steps": 35_000_000,
                "SPS": 1_500_000,
            },
            base_config,
        )

        rows = sweep_hypers.rank_wandb_runs(sweep_hypers.collect_wandb_runs(root, project="df38"))

    if [row["run"] for row in rows] != ["run-high", "run-low"]:
        failures.append(f"expected high-target run first, got {[row['run'] for row in rows]!r}")
    high = rows[0]
    if high["target"] != 8.9 or high["hidden_size"] != 128 or high["num_layers"] != 3:
        failures.append(f"high run fields not extracted correctly: {high!r}")
    if not math.isclose(high["surface_saturation"], 0.3, abs_tol=1e-9):
        failures.append(f"surface saturation mismatch: {high['surface_saturation']!r}")


def test_benchmark_profiles_keep_current_and_historical_hypers(failures):
    if sweep_hypers.CURRENT_BENCHMARK_PROFILE != "df38_sane_50m":
        failures.append(f"unexpected current benchmark profile: {sweep_hypers.CURRENT_BENCHMARK_PROFILE!r}")

    current = sweep_hypers.benchmark_profile_overrides("df38_sane_50m")
    expected_current = {
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
    if current != expected_current:
        failures.append(f"current profile drifted from accepted df38 hypers: {current!r}")

    historical = sweep_hypers.benchmark_profile_overrides("pre_df38_ini_defaults")
    if historical.get("policy.num-layers") != "2":
        failures.append(f"historical profile should preserve old num_layers=2, got {historical.get('policy.num-layers')!r}")
    if historical.get("train.horizon") != "128":
        failures.append(f"historical profile should preserve old horizon=128, got {historical.get('train.horizon')!r}")
    if historical.get("train.ent-coef") != "0.1604920157441345":
        failures.append(f"historical profile should preserve old high entropy, got {historical.get('train.ent-coef')!r}")


def test_benchmark_trials_use_current_profile_as_baseline_by_seed(failures):
    trials = sweep_hypers.benchmark_trials(seeds=[101, 102])

    expected_names = ["baseline_000", "baseline_001"]
    if [name for name, _ in trials] != expected_names:
        failures.append(f"unexpected benchmark names: {[name for name, _ in trials]!r}")

    overrides = dict(trials)
    baseline = overrides["baseline_000"]
    expected_baseline = {"train.seed": "101"}
    expected_baseline.update(sweep_hypers.benchmark_profile_overrides("df38_sane_50m"))
    if baseline != expected_baseline:
        failures.append(f"baseline should pin current accepted profile, got {baseline!r}")


def test_benchmark_trials_can_include_historical_candidate_by_seed(failures):
    trials = sweep_hypers.benchmark_trials(
        seeds=[101, 102],
        candidate_profile="pre_df38_ini_defaults",
    )

    expected_names = ["baseline_000", "baseline_001", "candidate_000", "candidate_001"]
    if [name for name, _ in trials] != expected_names:
        failures.append(f"unexpected benchmark names: {[name for name, _ in trials]!r}")

    overrides = dict(trials)
    candidate = overrides["candidate_000"]
    expected_candidate = {
        "train.seed": "101",
        "policy.hidden-size": "128",
        "policy.num-layers": "2",
        "train.horizon": "128",
        "train.learning-rate": "0.01788370841005079",
        "train.ent-coef": "0.1604920157441345",
        "train.clip-coef": "0.41000268739635415",
    }
    for key, value in expected_candidate.items():
        if candidate.get(key) != value:
            failures.append(f"candidate override {key} expected {value!r}, got {candidate.get(key)!r}")


def test_change_benchmark_trials_pin_full_profile_by_seed(failures):
    trials = sweep_hypers.change_benchmark_trials(seeds=[101, 102])

    expected_names = ["change_000", "change_001"]
    if [name for name, _ in trials] != expected_names:
        failures.append(f"unexpected change benchmark names: {[name for name, _ in trials]!r}")

    overrides = dict(trials)
    expected = {"train.seed": "101"}
    expected.update(sweep_hypers.FIXED_BENCHMARK_HYPERS)
    if overrides["change_000"] != expected:
        failures.append(f"change benchmark should pin full profile, got {overrides['change_000']!r}")
    if overrides["change_001"].get("train.seed") != "102":
        failures.append(f"second change run should use paired seed 102, got {overrides['change_001']!r}")


def test_benchmark_comparison_matches_or_beats_baseline(failures):
    rows = [
        {"trial": "baseline_000", "status": "ok", "max_target": "7.9", "selection_score": "7.0"},
        {"trial": "baseline_001", "status": "ok", "max_target": "8.9", "selection_score": "8.0"},
        {"trial": "candidate_000", "status": "ok", "max_target": "7.9", "selection_score": "6.4"},
        {"trial": "candidate_001", "status": "ok", "max_target": "8.9", "selection_score": "7.2"},
    ]

    result = sweep_hypers.compare_benchmark_rows(rows)
    if result["passed"] is not True:
        failures.append(f"expected matching sane benchmark to pass: {result!r}")

    regressed = [
        {"trial": "baseline_000", "status": "ok", "max_target": "7.9", "selection_score": "7.0"},
        {"trial": "baseline_001", "status": "ok", "max_target": "8.9", "selection_score": "8.0"},
        {"trial": "candidate_000", "status": "failed:1", "max_target": "1.9", "selection_score": "1.0"},
        {"trial": "candidate_001", "status": "ok", "max_target": "1.9", "selection_score": "1.0"},
    ]
    result = sweep_hypers.compare_benchmark_rows(regressed)
    if result["passed"] is not False:
        failures.append("expected regressed sane benchmark to fail")
    if not any("median max_target" in failure for failure in result["failures"]):
        failures.append(f"expected median target failure, got {result['failures']!r}")
    if not any("failed runs" in failure for failure in result["failures"]):
        failures.append(f"expected failed-run failure, got {result['failures']!r}")


def test_change_benchmark_comparison_requires_matching_overrides(failures):
    old_overrides = json.dumps({"train.seed": "101", "train.learning-rate": "0.024"}, sort_keys=True)
    new_overrides = json.dumps({"train.seed": "101", "train.learning-rate": "0.024"}, sort_keys=True)
    old_rows = [
        {
            "trial": "change_000",
            "status": "ok",
            "max_target": "7.9",
            "selection_score": "7.0",
            "overrides": old_overrides,
        },
        {
            "trial": "change_001",
            "status": "ok",
            "max_target": "8.9",
            "selection_score": "8.0",
            "overrides": json.dumps({"train.seed": "102", "train.learning-rate": "0.024"}, sort_keys=True),
        },
    ]
    new_rows = [
        {
            "trial": "change_000",
            "status": "ok",
            "max_target": "7.9",
            "selection_score": "6.4",
            "overrides": new_overrides,
        },
        {
            "trial": "change_001",
            "status": "ok",
            "max_target": "8.9",
            "selection_score": "7.2",
            "overrides": json.dumps({"train.seed": "102", "train.learning-rate": "0.024"}, sort_keys=True),
        },
    ]

    result = sweep_hypers.compare_change_benchmark_rows(old_rows, new_rows)
    if result["passed"] is not True:
        failures.append(f"expected matching change benchmark to pass: {result!r}")

    mismatched = [dict(row) for row in new_rows]
    mismatched[0]["overrides"] = json.dumps(
        {"train.seed": "101", "train.learning-rate": "0.030"},
        sort_keys=True,
    )
    result = sweep_hypers.compare_change_benchmark_rows(old_rows, mismatched)
    if result["passed"] is not False:
        failures.append("expected override mismatch to fail change benchmark")
    if not any("overrides differ" in failure for failure in result["failures"]):
        failures.append(f"expected override mismatch failure, got {result['failures']!r}")

    regressed = [dict(row) for row in new_rows]
    regressed[1]["max_target"] = "1.9"
    regressed[1]["selection_score"] = "1.0"
    result = sweep_hypers.compare_change_benchmark_rows(old_rows, regressed)
    if result["passed"] is not False:
        failures.append("expected regressed change benchmark to fail")
    if not any("candidate median max_target" in failure for failure in result["failures"]):
        failures.append(f"expected median target failure, got {result['failures']!r}")


def test_dogfight_protein_targets_soft_curriculum_quality(failures):
    import pufferlib.pufferl as pufferl

    old_argv = sys.argv
    try:
        sys.argv = ["pufferl.py"]
        args = pufferl.load_config("dogfight")
    finally:
        sys.argv = old_argv

    metric = args["sweep"]["metric"]
    if metric != "curriculum_soft_quality":
        failures.append(f"expected Protein metric curriculum_soft_quality, got {metric!r}")

    min_steps = args["sweep"].get("early_stop_min_steps")
    if min_steps != 40_000_000:
        failures.append(f"expected early_stop_min_steps 40M, got {min_steps!r}")


def test_dogfight_protein_sweep_excludes_pathological_model_sizes(failures):
    import pufferlib.pufferl as pufferl

    old_argv = sys.argv
    try:
        sys.argv = ["pufferl.py"]
        args = pufferl.load_config("dogfight")
    finally:
        sys.argv = old_argv

    sweep = args["sweep"]
    if args["train"]["total_timesteps"] != 50_000_000:
        failures.append(
            "default train.total_timesteps should match the 50M promotion-probe center, "
            f"got {args['train']['total_timesteps']!r}"
        )
    if sweep["policy"]["hidden_size"]["max"] > 256:
        failures.append(f"hidden_size max should be <=256, got {sweep['policy']['hidden_size']['max']!r}")
    if sweep["policy"]["num_layers"]["max"] > 4:
        failures.append(f"num_layers max should be <=4, got {sweep['policy']['num_layers']['max']!r}")
    if sweep["train"]["horizon"]["min"] < 32:
        failures.append(f"horizon min should be >=32, got {sweep['train']['horizon']['min']!r}")
    if sweep["train"]["total_timesteps"]["min"] != 25_000_000:
        failures.append(
            "total_timesteps min should target 25M promotion probes, "
            f"got {sweep['train']['total_timesteps']['min']!r}"
        )
    if sweep["train"]["total_timesteps"]["mean"] != 50_000_000:
        failures.append(
            "total_timesteps mean should target 50M promotion probes, "
            f"got {sweep['train']['total_timesteps']['mean']!r}"
        )
    if sweep["train"]["total_timesteps"]["max"] != 100_000_000:
        failures.append(
            "total_timesteps max should target 100M promotion probes, "
            f"got {sweep['train']['total_timesteps']['max']!r}"
        )


TESTS = [
    test_surface_saturation_score_accepts_unsaturated_run,
    test_surface_saturation_rejects_saturated_run,
    test_surface_saturation_rejects_missing_required_metrics,
    test_startup_dashboard_nan_is_not_an_error,
    test_base_stage_kill_rate_derives_from_kills_and_eps,
    test_random_trials_include_entropy_annealing_space,
    test_random_trial_overrides_parse_as_dogfight_cli,
    test_trials_from_summary_selects_top_non_rejected,
    test_wandb_analysis_ranks_high_targets_and_extracts_hypers,
    test_benchmark_profiles_keep_current_and_historical_hypers,
    test_benchmark_trials_use_current_profile_as_baseline_by_seed,
    test_benchmark_trials_can_include_historical_candidate_by_seed,
    test_change_benchmark_trials_pin_full_profile_by_seed,
    test_benchmark_comparison_matches_or_beats_baseline,
    test_change_benchmark_comparison_requires_matching_overrides,
    test_dogfight_protein_targets_soft_curriculum_quality,
    test_dogfight_protein_sweep_excludes_pathological_model_sizes,
]


def main():
    failures = []
    for test in TESTS:
        before = len(failures)
        try:
            test(failures)
        except Exception as exc:
            failures.append(f"{test.__name__}: {type(exc).__name__}: {exc}")
        status = "OK" if len(failures) == before else "FAIL"
        print(f"  {test.__name__:55s} [{status}]")

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"\nsweep_hypers: {len(TESTS)}/{len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
