#include <stdio.h>
#include <stdlib.h>

#include "test_common.h"

static int test_stage0_max_steps_not_sticky(void) {
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

    int saw_extended = 0;
    int checked_normal_after_extended = 0;
    for (int i = 0; i < 1000; i++) {
        c_reset(&t.env);
        if (t.env.player.pos.z == 400.0f) {
            saw_extended = 1;
            if (t.env.max_steps != 2000) {
                printf("stage0_max_steps: low-alt max_steps=%d [FAIL]\n", t.env.max_steps);
                return 1;
            }
        } else if (saw_extended) {
            checked_normal_after_extended = 1;
            if (t.env.max_steps != 300) {
                printf("stage0_max_steps: normal max_steps stuck at %d [FAIL]\n", t.env.max_steps);
                return 1;
            }
            break;
        }
    }

    if (!checked_normal_after_extended) {
        printf("stage0_max_steps: did not sample normal after low-alt variant [FAIL]\n");
        return 1;
    }

    printf("stage0_max_steps: normal resets restore configured max_steps [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    return test_stage0_max_steps_not_sticky();
}
