#include <math.h>
#include <stdio.h>
#include <string.h>

#include "test_common.h"

static void setup_curriculum_env(TestEnv* t, int stage) {
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
}

static int altitude_in_bounds(float alt) {
    return alt >= 300.0f && alt <= 4700.0f;
}

static int test_stage_8_17_spawn_safety_and_caps(void) {
    TestEnv t;
    for (int stage = CURRICULUM_SIDE_FAR; stage <= CURRICULUM_HARD_MANEUVERING; stage++) {
        setup_curriculum_env(&t, stage);

        for (int i = 0; i < 64; i++) {
            c_reset(&t.env);
            if (t.env.stage != stage || t.env.max_steps != STAGES[stage].max_steps) {
                printf("stage%d_spawn_caps: stage=%d max_steps=%d expected=%d [FAIL]\n",
                    stage, t.env.stage, t.env.max_steps, STAGES[stage].max_steps);
                return 1;
            }
            if (!altitude_in_bounds(t.env.player.pos.z)
                    || !altitude_in_bounds(t.env.opponent.pos.z)) {
                printf("stage%d_spawn_alt: player=%.1f opponent=%.1f [FAIL]\n",
                    stage, t.env.player.pos.z, t.env.opponent.pos.z);
                return 1;
            }

            const float neutral[TEST_NUM_ATNS] = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f};
            for (int step = 0; step < 50; step++) {
                memcpy(t.env.actions, neutral, sizeof(neutral));
                c_step(&t.env);
                if (t.env.terminals[0] != 0.0f || t.env.death_reason != DEATH_NONE) {
                    printf("stage%d_neutral_trace: terminal=%.0f reason=%d step=%d [FAIL]\n",
                        stage, t.env.terminals[0], t.env.death_reason, step);
                    return 1;
                }
                if (t.env.player.pos.z < 100.0f || t.env.opponent.pos.z < 100.0f) {
                    printf("stage%d_neutral_ground: player=%.1f opponent=%.1f step=%d [FAIL]\n",
                        stage, t.env.player.pos.z, t.env.opponent.pos.z, step);
                    return 1;
                }
            }
        }
    }

    printf("stage8_17_spawn_safety: caps, altitude, short neutral traces [OK]\n");
    return 0;
}

static int test_stage8_9_side_geometry(void) {
    TestEnv t;
    for (int stage = CURRICULUM_SIDE_FAR; stage <= CURRICULUM_SIDE_MANEUVERING; stage++) {
        int standard_spawns = 0;
        int energy_spawns = 0;
        setup_curriculum_env(&t, stage);

        for (int i = 0; i < 200; i++) {
            c_reset(&t.env);
            Vec3 rel = sub3(t.env.opponent.pos, t.env.player.pos);
            float az_deg = fabsf(atan2f(rel.y, rel.x) * RAD);
            float alt_delta = rel.z;
            float player_speed = norm3(t.env.player.vel);
            float opponent_speed = norm3(t.env.opponent.vel);

            if (opponent_speed < player_speed * 0.65f) {
                energy_spawns++;
                if (alt_delta < -1.0f) {
                    printf("stage%d_energy_spawn: alt_delta=%.1f player_v=%.1f opp_v=%.1f [FAIL]\n",
                        stage, alt_delta, player_speed, opponent_speed);
                    return 1;
                }
                continue;
            }

            standard_spawns++;
            if (az_deg < STAGES[stage].angle_min_deg - 1.0f
                    || az_deg > STAGES[stage].angle_max_deg + 1.0f) {
                printf("stage%d_side_az: az=%.1f expected=[%.0f,%.0f] [FAIL]\n",
                    stage, az_deg, STAGES[stage].angle_min_deg,
                    STAGES[stage].angle_max_deg);
                return 1;
            }
        }

        if (standard_spawns == 0 || energy_spawns == 0) {
            printf("stage%d_side_mix: standard=%d energy=%d [FAIL]\n",
                stage, standard_spawns, energy_spawns);
            return 1;
        }
    }

    printf("stage8_9_side_geometry: side-angle and energy-building spawns [OK]\n");
    return 0;
}

static int test_stage10_12_vertical_advantage_geometry(void) {
    TestEnv t;

    setup_curriculum_env(&t, CURRICULUM_DIVE_ATTACK);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.player.pos.z - t.env.opponent.pos.z;
        if (alt_delta < 430.0f || alt_delta > 570.0f) {
            printf("stage10_dive_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    setup_curriculum_env(&t, CURRICULUM_ZOOM_ATTACK);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.opponent.pos.z - t.env.player.pos.z;
        if (alt_delta < 250.0f || alt_delta > 350.0f) {
            printf("stage11_zoom_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    setup_curriculum_env(&t, CURRICULUM_REAR_CHASE);
    for (int i = 0; i < 100; i++) {
        c_reset(&t.env);
        float alt_delta = t.env.player.pos.z - t.env.opponent.pos.z;
        if (alt_delta < 430.0f || alt_delta > 570.0f) {
            printf("stage12_rear_alt_delta: %.1f [FAIL]\n", alt_delta);
            return 1;
        }
    }

    printf("stage10_12_vertical_geometry: altitude-advantage spawns [OK]\n");
    return 0;
}

int main(void) {
    srand(42);
    int fails = 0;
    fails += test_stage_8_17_spawn_safety_and_caps();
    fails += test_stage8_9_side_geometry();
    fails += test_stage10_12_vertical_advantage_geometry();
    return fails;
}
