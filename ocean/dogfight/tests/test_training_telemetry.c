#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test_common.h"

static int nearly(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

static int setup_curriculum_env(TestEnv* t, int stage) {
    memset(t, 0, sizeof(*t));
    t->env.num_agents = 1;
    t->env.max_steps = 300;
    t->env.rng = 42;
    t->env.observations = t->observations;
    t->env.actions = t->actions;
    t->env.rewards = t->rewards;
    t->env.terminals = t->terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t->env, 1, &rcfg, 1, 0, 0);
    set_curriculum_stage(&t->env, stage);
    return 0;
}

static int test_stage_has_no_low_alt_variant(int stage, int expected_max_steps) {
    TestEnv t;
    setup_curriculum_env(&t, stage);

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.low_altitude_variant || t.env.max_steps != expected_max_steps) {
            printf("low_alt_stage%d: low_alt=%d max_steps=%d [FAIL]\n",
                    stage, t.env.low_altitude_variant, t.env.max_steps);
            return 1;
        }
    }

    printf("low_alt_stage%d: disabled in early curriculum [OK]\n", stage);
    return 0;
}

static int test_low_alt_variant_log_for_stage(int stage) {
    TestEnv t;
    setup_curriculum_env(&t, stage);

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (!t.env.low_altitude_variant) continue;

        if (t.env.max_steps != 2000) {
            printf("low_alt_stage%d: max_steps=%d [FAIL]\n", stage, t.env.max_steps);
            return 1;
        }

        t.env.tick = t.env.max_steps;
        t.env.death_reason = DEATH_TIMEOUT;
        add_log(&t.env);

        if (!nearly(t.env.log.low_alt_variant_eps, 1.0f) ||
                !nearly(t.env.log.low_alt_variant_ticks, 2000.0f)) {
            printf("low_alt_stage%d: eps=%.1f ticks=%.1f [FAIL]\n",
                    stage, t.env.log.low_alt_variant_eps,
                    t.env.log.low_alt_variant_ticks);
            return 1;
        }

        printf("low_alt_stage%d: low-alt episode/tick telemetry [OK]\n", stage);
        return 0;
    }

    printf("low_alt_stage%d: did not sample low-alt variant [FAIL]\n", stage);
    return 1;
}

static int test_action_telemetry(void) {
    TestEnv t;
    setup_env(&t, 1);

    const float a[TEST_NUM_ATNS] = {0.0f, 0.50f, -1.0f, 0.25f, 1.0f};
    run_steps(&t, 4, a);
    t.env.death_reason = DEATH_TIMEOUT;
    add_log(&t.env);

    int fail = 0;
    fail |= !nearly(t.env.log.action_abs_elevator, 0.50f);
    fail |= !nearly(t.env.log.action_abs_aileron, 1.00f);
    fail |= !nearly(t.env.log.action_abs_rudder, 0.25f);
    fail |= !nearly(t.env.log.action_abs_trigger, 1.00f);
    fail |= !nearly(t.env.log.action_sat_elevator, 0.00f);
    fail |= !nearly(t.env.log.action_sat_aileron, 1.00f);
    fail |= !nearly(t.env.log.action_sat_rudder, 0.00f);
    fail |= !nearly(t.env.log.action_sat_trigger, 1.00f);

    if (fail) {
        printf("action_telemetry: abs=(%.2f %.2f %.2f %.2f) sat=(%.2f %.2f %.2f %.2f) [FAIL]\n",
                t.env.log.action_abs_elevator,
                t.env.log.action_abs_aileron,
                t.env.log.action_abs_rudder,
                t.env.log.action_abs_trigger,
                t.env.log.action_sat_elevator,
                t.env.log.action_sat_aileron,
                t.env.log.action_sat_rudder,
                t.env.log.action_sat_trigger);
        return 1;
    }

    printf("action_telemetry: per-action mean abs and saturation [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += test_stage_has_no_low_alt_variant(0, 300);
    fails += test_stage_has_no_low_alt_variant(1, 500);
    fails += test_low_alt_variant_log_for_stage(3);
    fails += test_action_telemetry();
    return fails;
}
