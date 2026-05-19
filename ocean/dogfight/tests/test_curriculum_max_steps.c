#include <stdio.h>
#include <stdlib.h>

#include "test_common.h"

static int test_stage0_max_steps_stays_configured(void) {
    TestEnv t;
    memset(&t, 0, sizeof(t));
    t.env.num_agents = 1;
    t.env.max_steps = 300;
    t.env.rng = 42;
    t.env.observations = t.observations;
    t.env.actions = t.actions;
    t.env.rewards = t.rewards;
    t.env.terminals = t.terminals;

    RewardConfig rcfg = test_default_rcfg();
    init(&t.env, 1, &rcfg, 1, 0, 0);
    set_curriculum_target(&t.env, 0.0f);

    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.low_altitude_variant || t.env.max_steps != 300) {
            printf("stage0_max_steps: low_alt=%d max_steps=%d [FAIL]\n",
                    t.env.low_altitude_variant, t.env.max_steps);
            return 1;
        }
    }

    printf("stage0_max_steps: early curriculum keeps configured max_steps [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    return test_stage0_max_steps_stays_configured();
}
