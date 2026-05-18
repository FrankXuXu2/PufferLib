"""Unit tests for the Dogfight training curriculum controller."""

import os
import sys


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)


class FakeBackend:
    def __init__(self):
        self.targets = []

    def set_curriculum_target(self, pufferl, target):
        self.targets.append(float(target))


def main():
    from pufferlib.pufferl import setup_curriculum, step_curriculum

    args = {
        "env": {"curriculum_enabled": 1},
        "curriculum": {
            "enabled": 1,
            "initial_target": 0.9,
            "max_target": 2.0,
            "step": 1.0,
            "promote_threshold": 0.9,
            "min_episodes": 100,
        },
    }
    backend = FakeBackend()
    state = setup_curriculum(args, backend, object())
    assert state["target"] == 0.9
    assert backend.targets == [0.9]

    flat_logs = {
        "env/n": 200.0,
        "env/base_stage_eps": 0.5,
        "env/base_stage_kills": 0.4,
    }
    step_curriculum(state, backend, object(), flat_logs, epoch=1)
    assert state["target"] == 0.9
    assert backend.targets == [0.9]
    assert abs(flat_logs["env/base_stage_kill_rate"] - 0.8) < 1e-6

    flat_logs = {
        "env/n": 200.0,
        "env/base_stage_eps": 0.5,
        "env/base_stage_kills": 0.46,
    }
    step_curriculum(state, backend, object(), flat_logs, epoch=2)
    assert state["target"] == 1.9
    assert backend.targets == [0.9, 1.9]
    assert flat_logs["env/curriculum_target"] == 1.9

    step_curriculum(state, backend, object(), flat_logs, epoch=3)
    assert state["target"] == 2.0
    step_curriculum(state, backend, object(), flat_logs, epoch=4)
    assert state["target"] == 2.0
    assert backend.targets == [0.9, 1.9, 2.0]

    print("  test_curriculum_progress          [OK]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
