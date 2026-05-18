#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "test_common.h"

static int nearf(float a, float b) {
    return fabsf(a - b) < 1e-6f;
}

static int test_runtime_config_applies_selfplay_knobs(void) {
    TestEnv t;
    memset(&t, 0, sizeof(t));
    t.env.num_agents = 1;
    t.env.max_steps = 300;
    t.env.rng = 7;
    t.env.observations = t.observations;
    t.env.actions = t.actions;
    t.env.rewards = t.rewards;
    t.env.terminals = t.terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t.env, 1, &rcfg, 1, 0, 0);

    RuntimeConfig cfg = {
        .eval_spawn_mode = 2,
        .recovery_enabled = 1,
        .recovery_altitude_threshold = 625.0f,
        .recovery_trigger_prob = 0.25f,
        .recovery_speed_threshold = 82.0f,
        .recovery_bank_deg = 45.0f,
        .domain_randomization = 0.05f,
        .vertical_spawn_prob = 0.20f,
    };
    apply_runtime_config(&t.env, &cfg);

    if (t.env.eval_spawn_mode != 2 ||
            !nearf(t.env.recovery_altitude_threshold, 625.0f) ||
            !nearf(t.env.recovery_trigger_prob, 0.25f) ||
            !nearf(t.env.recovery_speed_threshold, 82.0f) ||
            !nearf(t.env.recovery_bank_deg, 45.0f) ||
            !nearf(t.env.domain_randomization, 0.05f) ||
            !nearf(t.env.vertical_spawn_prob, 0.20f)) {
        printf("runtime_config: self-play knobs not applied [FAIL]\n");
        return 1;
    }

    cfg.recovery_enabled = 0;
    apply_runtime_config(&t.env, &cfg);
    if (!(t.env.recovery_altitude_threshold < 0.0f)) {
        printf("runtime_config: recovery disable did not set negative threshold [FAIL]\n");
        return 1;
    }

    printf("runtime_config: self-play/domain knobs applied [OK]\n");
    return 0;
}

int main(void) {
    srand(7);
    return test_runtime_config_applies_selfplay_knobs();
}
