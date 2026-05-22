"""Unit tests for Protein early-stop log shape handling."""

import math
import os
import queue
import sys
import types
from collections import deque
from copy import deepcopy


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)


def _protein_probe():
    from pufferlib.sweep import Protein

    probe = Protein.__new__(Protein)
    probe._running_target_buffer = deque(maxlen=30)
    probe.metric_distribution = "linear"
    probe.observed = []

    def get_early_stop_threshold(cost):
        probe.observed.append(("threshold", cost))
        return 2.0

    def should_stop(score, cost):
        probe.observed.append(("score", score, cost))
        return score < 2.0

    probe.get_early_stop_threshold = get_early_stop_threshold
    probe.should_stop = should_stop
    return probe


def _assert_curriculum_metric_seen(logs):
    probe = _protein_probe()
    assert probe.early_stop(logs, "env/curriculum_target") is True
    assert ("score", 1.9, 1.0) in probe.observed
    assert logs["early_stop_threshold"] == 2.0
    assert logs["is_loss_nan"] is False


def test_protein_early_stop_accepts_flat_and_nested_logs():
    _assert_curriculum_metric_seen({
        "env/curriculum_target": 1.9,
        "loss/policy": 0.0,
        "uptime": 1.0,
    })
    _assert_curriculum_metric_seen({
        "env": {"curriculum_target": 1.9},
        "loss": {"policy": 0.0},
        "uptime": 1.0,
    })

    flat_nan_logs = {
        "env/curriculum_target": 3.0,
        "loss/policy": math.nan,
        "uptime": 1.0,
    }
    assert _protein_probe().early_stop(flat_nan_logs, "env/curriculum_target") is True
    assert flat_nan_logs["is_loss_nan"] is True

    nested_nan_logs = {
        "env": {"curriculum_target": 3.0},
        "loss": {"policy": math.nan},
        "uptime": 1.0,
    }
    assert _protein_probe().early_stop(nested_nan_logs, "env/curriculum_target") is True
    assert nested_nan_logs["is_loss_nan"] is True


def test_downsample_sweep_logs_accepts_late_protein_keys():
    from pufferlib.pufferl import downsample_sweep_logs

    metrics = downsample_sweep_logs([
        {
            "agent_steps": 1,
            "uptime": 1.0,
            "env/curriculum_target": 1.0,
        },
        {
            "agent_steps": 2,
            "uptime": 2.0,
            "env/curriculum_target": 2.0,
        },
        {
            "agent_steps": 3,
            "uptime": 3.0,
            "env/curriculum_target": 3.0,
            "early_stop_threshold": 0.5,
            "is_loss_nan": False,
        },
        {
            "agent_steps": 4,
            "uptime": 4.0,
            "env/curriculum_target": 4.0,
            "early_stop_threshold": 0.75,
            "is_loss_nan": False,
        },
    ], n=3)
    assert metrics["env/curriculum_target"] == [1.5, 3.5, 4.0]
    assert math.isnan(metrics["early_stop_threshold"][0])
    assert metrics["early_stop_threshold"][1:] == [0.625, 0.75]
    assert math.isnan(metrics["is_loss_nan"][0])
    assert metrics["is_loss_nan"][1:] == [0.0, False]


def test_dogfight_soft_quality_is_derived_from_existing_logs():
    import pufferlib.pufferl as pufferl

    logs = {
        "env/curriculum_target": 9.0,
        "env/action_sat_elevator": 0.0,
        "env/action_sat_aileron": 0.5,
        "env/action_sat_rudder": 1.0,
    }
    pufferl.add_derived_sweep_metrics({
        "env_name": "dogfight",
        "curriculum": {"max_target": 18.0},
    }, logs)

    assert math.isclose(logs["env/curriculum_soft_quality"], 0.375)
    assert "env/curriculum_quality" not in logs


def test_sweep_early_stop_respects_min_step_floor():
    from pufferlib.pufferl import sweep_early_stop_ready

    sweep_config = {"early_stop_min_steps": 40_000_000}
    assert sweep_early_stop_ready(39_999_999, 50_000_000, sweep_config) is False
    assert sweep_early_stop_ready(40_000_001, 50_000_000, sweep_config) is True
    assert sweep_early_stop_ready(25_000_000, 25_000_000, sweep_config) is False

    assert sweep_early_stop_ready(10_000_001, 50_000_000, {}) is True


def _sweep_args(max_runs):
    return {
        "env_name": "dogfight",
        "train": {
            "gpus": 1,
            "total_timesteps": 10,
            "horizon": 1,
            "minibatch_size": 1,
        },
        "vec": {
            "total_agents": 1,
            "num_threads": 4,
        },
        "sweep": {
            "method": "Protein",
            "metric": "curriculum_target",
            "metric_distribution": "linear",
            "goal": "maximize",
            "max_runs": max_runs,
            "gpus": 4,
            "downsample": 1,
            "train": {
                "total_timesteps": {
                    "min": 10,
                    "max": 10,
                },
            },
        },
    }


class _FakeSweep:
    instances = []

    def __init__(self, config):
        self.config = config
        self.observed = []
        self.suggested = 0
        _FakeSweep.instances.append(self)

    def suggest(self, args, fixed_total_timesteps=None):
        self.suggested += 1

    def observe(self, args, score, cost, is_failure=False):
        self.observed.append((score, cost, is_failure, args["train"]["total_timesteps"]))


class _ExitedProcess:
    exitcode = 1

    def __init__(self):
        self.terminated = False
        self.joined = 0

    def is_alive(self):
        return False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        self.joined += 1


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def get(self, timeout=None):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)


class _FakeContext:
    def Queue(self):
        return _FakeQueue()


def _run_sweep_with_fake_train(args, fake_train, wait_seconds=0.05):
    import pufferlib.pufferl as pufferl
    import pufferlib.sweep as sweep_mod

    original_train = pufferl.train
    original_protein = sweep_mod.Protein
    original_wait = pufferl.SWEEP_RESULT_WAIT_SECONDS
    original_get_context = pufferl.mp.get_context
    original_resolve_backend = pufferl._resolve_backend
    _FakeSweep.instances = []
    try:
        pufferl.train = fake_train
        pufferl.SWEEP_RESULT_WAIT_SECONDS = wait_seconds
        pufferl.mp.get_context = lambda method=None: _FakeContext()
        pufferl._resolve_backend = lambda args: object()
        sweep_mod.Protein = _FakeSweep
        pufferl.sweep("dogfight", args=deepcopy(args))
        return _FakeSweep.instances[-1]
    finally:
        pufferl.train = original_train
        pufferl.SWEEP_RESULT_WAIT_SECONDS = original_wait
        pufferl.mp.get_context = original_get_context
        pufferl._resolve_backend = original_resolve_backend
        sweep_mod.Protein = original_protein


def test_sweep_drains_final_active_trials():
    launched = []

    def fake_train(env_name, exp_args, gpus, **kwargs):
        gpu_id = list(gpus)[0]
        launched.append(gpu_id)
        kwargs["result_queue"].put((gpu_id, [1.0], [2.0], [3]))
        return []

    sweep = _run_sweep_with_fake_train(_sweep_args(max_runs=2), fake_train)
    assert launched == [0, 1]
    assert len(sweep.observed) == 2
    assert all(not row[2] for row in sweep.observed)


def test_sweep_marks_exited_worker_without_result_failed():
    launched = []

    def fake_train(env_name, exp_args, gpus, **kwargs):
        gpu_id = list(gpus)[0]
        launched.append(gpu_id)
        if len(launched) == 1:
            return [_ExitedProcess()]
        kwargs["result_queue"].put((gpu_id, [1.0], [2.0], [3]))
        return []

    sweep = _run_sweep_with_fake_train(
        _sweep_args(max_runs=1), fake_train, wait_seconds=0)
    assert launched == [0]
    assert sweep.observed[0][2] is True


def test_sweep_counts_no_metric_failures_against_max_runs():
    launched = []

    def fake_train(env_name, exp_args, gpus, **kwargs):
        gpu_id = list(gpus)[0]
        launched.append(gpu_id)
        if len(launched) > 2:
            raise AssertionError("sweep launched more trials than max_runs")
        kwargs["result_queue"].put((gpu_id, None, None, None))
        return []

    sweep = _run_sweep_with_fake_train(
        _sweep_args(max_runs=2), fake_train, wait_seconds=0)
    assert launched == [0, 1]
    assert len(sweep.observed) == 2
    assert all(row[2] for row in sweep.observed)


def test_finish_wandb_run_finishes_active_run():
    import pufferlib.pufferl as pufferl

    class _Run:
        def __init__(self):
            self.finished = 0

        def finish(self):
            self.finished += 1

    run = _Run()
    fake_wandb = types.SimpleNamespace(run=run)
    sentinel = object()
    original = sys.modules.get("wandb", sentinel)
    try:
        sys.modules["wandb"] = fake_wandb
        pufferl._finish_wandb_run({"wandb": True})
        assert run.finished == 1

        fake_wandb.run = None
        pufferl._finish_wandb_run({"wandb": True})
        pufferl._finish_wandb_run({"wandb": False})
        assert run.finished == 1
    finally:
        if original is sentinel:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = original


def main():
    test_protein_early_stop_accepts_flat_and_nested_logs()
    test_downsample_sweep_logs_accepts_late_protein_keys()
    test_dogfight_soft_quality_is_derived_from_existing_logs()
    test_sweep_early_stop_respects_min_step_floor()
    test_sweep_drains_final_active_trials()
    test_sweep_marks_exited_worker_without_result_failed()
    test_sweep_counts_no_metric_failures_against_max_runs()
    test_finish_wandb_run_finishes_active_run()

    print("  test_sweep_early_stop_logs        [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
