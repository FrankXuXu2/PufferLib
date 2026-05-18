#include <math.h>
#include <stdio.h>

#include "test_common.h"

static int test_unbounded_native_actions_are_clamped(void) {
    TestEnv t;
    setup_env(&t, 1);

    t.env.actions[0] = 3.0f;
    t.env.actions[1] = -4.0f;
    t.env.actions[2] = 5.0f;
    t.env.actions[3] = -6.0f;
    t.env.actions[4] = 7.0f;

    t_step(&t);

    int fail = 0;
    for (int i = 0; i < TEST_NUM_ATNS; i++) {
        if (t.env.actions[i] < -1.0f || t.env.actions[i] > 1.0f) {
            printf("action_bounds: action[%d]=%.2f outside Box(-1,1) [FAIL]\n",
                   i, t.env.actions[i]);
            fail = 1;
        }
    }

    if (fabsf(t.env.prev_elevator - t.env.actions[1]) > 1e-6f ||
        fabsf(t.env.prev_aileron - t.env.actions[2]) > 1e-6f ||
        fabsf(t.env.prev_rudder - t.env.actions[3]) > 1e-6f) {
        printf("action_bounds: previous controls stored unclamped values [FAIL]\n");
        fail = 1;
    }

    if (!isfinite(t.env.rewards[0]) || fabsf(t.env.rewards[0]) > 1.0f) {
        printf("action_bounds: reward=%.3f not finite/clamped [FAIL]\n", t.env.rewards[0]);
        fail = 1;
    }

    if (!fail) {
        printf("action_bounds: native continuous actions clamp to legacy Box(-1,1) [OK]\n");
    }
    return fail;
}

int main(void) {
    return test_unbounded_native_actions_are_clamped();
}
