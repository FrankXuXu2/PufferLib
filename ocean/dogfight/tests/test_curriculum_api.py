"""Regression test for the native curriculum target setter."""

import os
import sys

import numpy as np


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, REPO)


def build_args():
    from pufferlib.pufferl import load_config

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        args = load_config("dogfight")
    finally:
        sys.argv = old_argv

    args["vec"]["total_agents"] = 16
    args["vec"]["num_buffers"] = 1
    args["vec"]["num_threads"] = 1
    args["env"]["curriculum_enabled"] = 1
    args["env"]["curriculum_randomize"] = 0
    return args


def main():
    from pufferlib import _C

    if getattr(_C, "env_name", None) != "dogfight":
        print("  test_curriculum_target_setter     [SKIP - _C not built for dogfight]")
        return 0

    assert hasattr(_C, "set_curriculum_target")

    args = build_args()
    vec = _C.create_vec(args, 0)
    try:
        vec.set_curriculum_target(3.0)
        vec.reset()

        actions = np.zeros((vec.total_agents, vec.num_atns), dtype=np.float32)
        for _ in range(args["env"]["max_steps"] + 2):
            vec.cpu_step(actions.ctypes.data)

        logs = vec.log()
        avg_stage = float(logs["avg_stage"])
        assert 2.9 <= avg_stage <= 3.1, logs
        print(f"  test_curriculum_target_setter     [OK] avg_stage={avg_stage:.2f}")
    finally:
        vec.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
