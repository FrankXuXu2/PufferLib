/*
 * Deterministic Dogfight episode harness for scriptable debugging.
 *
 * Build through scripts/debug_dogfight_episode.sh so debug artifacts stay in
 * /tmp/dogfight_debug instead of this source tree.
 */
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_common.h"

typedef enum ActionMode {
    ACTION_NEUTRAL = 0,
    ACTION_CONSTANT = 1,
} ActionMode;

typedef struct DebugArgs {
    int steps;
    unsigned int seed;
    int stage;
    int obs_scheme;
    ActionMode action_mode;
    float action_values[TEST_NUM_ATNS];
    const char* out_dir;
} DebugArgs;

__attribute__((noinline))
void dogfight_debug_anchor_reset(Dogfight* env) {
    (void)env;
}

__attribute__((noinline))
void dogfight_debug_anchor_pre_step(Dogfight* env, int step_index, float* action) {
    (void)env;
    (void)step_index;
    (void)action;
}

__attribute__((noinline))
void dogfight_debug_anchor_post_step(Dogfight* env, int step_index, float reward, float terminal) {
    (void)env;
    (void)step_index;
    (void)reward;
    (void)terminal;
}

__attribute__((noinline))
void dogfight_debug_anchor_terminal(Dogfight* env, int step_index, DeathReason reason) {
    (void)env;
    (void)step_index;
    (void)reason;
}

static void usage(const char* argv0) {
    fprintf(stderr,
        "Usage: %s [--steps N] [--seed N] [--stage N] [--obs-scheme N]\n"
        "          [--action neutral|constant]\n"
        "          [--action-values throttle,elevator,aileron,rudder,trigger]\n"
        "          [--out-dir DIR]\n",
        argv0);
}

static int parse_int(const char* s, int* out) {
    char* end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0') return 0;
    *out = (int)v;
    return 1;
}

static int parse_uint(const char* s, unsigned int* out) {
    char* end = NULL;
    errno = 0;
    unsigned long v = strtoul(s, &end, 10);
    if (errno || end == s || *end != '\0') return 0;
    *out = (unsigned int)v;
    return 1;
}

static int parse_action_values(const char* s, float out[TEST_NUM_ATNS]) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", s);
    char* save = NULL;
    char* tok = strtok_r(buf, ",", &save);
    for (int i = 0; i < TEST_NUM_ATNS; i++) {
        if (!tok) return 0;
        char* end = NULL;
        errno = 0;
        out[i] = strtof(tok, &end);
        if (errno || end == tok || *end != '\0') return 0;
        tok = strtok_r(NULL, ",", &save);
    }
    return tok == NULL;
}

static int parse_args(int argc, char** argv, DebugArgs* args) {
    *args = (DebugArgs){
        .steps = 300,
        .seed = 42,
        .stage = 0,
        .obs_scheme = 0,
        .action_mode = ACTION_NEUTRAL,
        .action_values = {0.5f, 0.0f, 0.0f, 0.0f, -1.0f},
        .out_dir = "/tmp/dogfight_debug",
    };

    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        const char* val = (i + 1 < argc) ? argv[i + 1] : NULL;
        if (strcmp(arg, "--steps") == 0 && val) {
            if (!parse_int(val, &args->steps)) return 0;
            i++;
        } else if (strcmp(arg, "--seed") == 0 && val) {
            if (!parse_uint(val, &args->seed)) return 0;
            i++;
        } else if (strcmp(arg, "--stage") == 0 && val) {
            if (!parse_int(val, &args->stage)) return 0;
            i++;
        } else if (strcmp(arg, "--obs-scheme") == 0 && val) {
            if (!parse_int(val, &args->obs_scheme)) return 0;
            i++;
        } else if (strcmp(arg, "--action") == 0 && val) {
            if (strcmp(val, "neutral") == 0) {
                args->action_mode = ACTION_NEUTRAL;
                memcpy(args->action_values, (float[TEST_NUM_ATNS]){0.5f, 0.0f, 0.0f, 0.0f, -1.0f}, sizeof(args->action_values));
            } else if (strcmp(val, "constant") == 0) {
                args->action_mode = ACTION_CONSTANT;
            } else {
                return 0;
            }
            i++;
        } else if (strcmp(arg, "--action-values") == 0 && val) {
            if (!parse_action_values(val, args->action_values)) return 0;
            args->action_mode = ACTION_CONSTANT;
            i++;
        } else if (strcmp(arg, "--out-dir") == 0 && val) {
            args->out_dir = val;
            i++;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            usage(argv[0]);
            exit(0);
        } else {
            return 0;
        }
    }

    return args->steps >= 0
        && args->stage >= 0 && args->stage < CURRICULUM_COUNT
        && args->obs_scheme >= 0 && args->obs_scheme < OBS_SCHEME_COUNT;
}

static const char* death_reason_name(DeathReason r) {
    switch (r) {
        case DEATH_NONE: return "NONE";
        case DEATH_KILL: return "KILL";
        case DEATH_OOB: return "OOB";
        case DEATH_TIMEOUT: return "TIMEOUT";
        case DEATH_SUPERSONIC: return "SUPERSONIC";
        default: return "UNKNOWN";
    }
}

static void write_csv_header(FILE* f) {
    fprintf(f,
        "step,tick,stage,reward,terminal,death_reason,"
        "action_throttle,action_elevator,action_aileron,action_rudder,action_trigger,"
        "player_x,player_y,player_z,player_vx,player_vy,player_vz,player_speed,player_g,"
        "opponent_x,opponent_y,opponent_z,opponent_vx,opponent_vy,opponent_vz,opponent_speed,"
        "distance,episode_return,kill,opp_kill\n");
}

static void write_csv_row(FILE* f, const Dogfight* env, int step_index) {
    Vec3 rel = sub3(env->opponent.pos, env->player.pos);
    fprintf(f,
        "%d,%d,%d,%.9g,%.0f,%d,"
        "%.9g,%.9g,%.9g,%.9g,%.9g,"
        "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
        "%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,"
        "%.9g,%.9g,%d,%d\n",
        step_index, env->tick, env->stage, env->rewards[0], env->terminals[0], env->death_reason,
        env->actions[0], env->actions[1], env->actions[2], env->actions[3], env->actions[4],
        env->player.pos.x, env->player.pos.y, env->player.pos.z,
        env->player.vel.x, env->player.vel.y, env->player.vel.z,
        norm3(env->player.vel), env->player.g_force,
        env->opponent.pos.x, env->opponent.pos.y, env->opponent.pos.z,
        env->opponent.vel.x, env->opponent.vel.y, env->opponent.vel.z,
        norm3(env->opponent.vel), norm3(rel), env->episode_return, env->kill, env->opp_kill);
}

static void write_summary(FILE* f, const DebugArgs* args, const Dogfight* env) {
    fprintf(f, "seed=%u\n", args->seed);
    fprintf(f, "steps=%d\n", args->steps);
    fprintf(f, "stage=%d\n", args->stage);
    fprintf(f, "obs_scheme=%d\n", args->obs_scheme);
    fprintf(f, "action_mode=%s\n", args->action_mode == ACTION_NEUTRAL ? "neutral" : "constant");
    fprintf(f, "action_values=%.9g,%.9g,%.9g,%.9g,%.9g\n",
        args->action_values[0], args->action_values[1], args->action_values[2],
        args->action_values[3], args->action_values[4]);
    fprintf(f, "final_tick=%d\n", env->tick);
    fprintf(f, "final_stage=%d\n", env->stage);
    fprintf(f, "final_reward=%.9g\n", env->rewards[0]);
    fprintf(f, "final_terminal=%.0f\n", env->terminals[0]);
    fprintf(f, "final_death_reason=%s (%d)\n", death_reason_name(env->death_reason), env->death_reason);
    fprintf(f, "episode_return=%.9g\n", env->episode_return);
    fprintf(f, "kill=%d\n", env->kill);
    fprintf(f, "opp_kill=%d\n", env->opp_kill);
}

int main(int argc, char** argv) {
    DebugArgs args;
    if (!parse_args(argc, argv, &args)) {
        usage(argv[0]);
        return 2;
    }

    char csv_path[512];
    char summary_path[512];
    snprintf(csv_path, sizeof(csv_path), "%s/episode.csv", args.out_dir);
    snprintf(summary_path, sizeof(summary_path), "%s/summary.txt", args.out_dir);

    FILE* csv = fopen(csv_path, "w");
    if (!csv) {
        fprintf(stderr, "ERROR: failed to open %s\n", csv_path);
        return 2;
    }
    FILE* summary = fopen(summary_path, "w");
    if (!summary) {
        fprintf(stderr, "ERROR: failed to open %s\n", summary_path);
        fclose(csv);
        return 2;
    }

    srand(args.seed);

    TestEnv t;
    setup_env(&t, args.obs_scheme);
    t.env.curriculum_enabled = 1;
    t.env.curriculum_randomize = 0;
    set_curriculum_stage(&t.env, args.stage);
    c_reset(&t.env);
    dogfight_debug_anchor_reset(&t.env);

    write_csv_header(csv);
    write_csv_row(csv, &t.env, 0);

    for (int step = 1; step <= args.steps; step++) {
        memcpy(t.env.actions, args.action_values, sizeof(args.action_values));
        dogfight_debug_anchor_pre_step(&t.env, step, t.env.actions);
        c_step(&t.env);
        dogfight_debug_anchor_post_step(&t.env, step, t.env.rewards[0], t.env.terminals[0]);
        write_csv_row(csv, &t.env, step);
        if (t.env.terminals[0] != 0.0f) {
            dogfight_debug_anchor_terminal(&t.env, step, t.env.death_reason);
        }
    }

    write_summary(summary, &args, &t.env);
    fclose(csv);
    fclose(summary);

    printf("dogfight_debug_episode: wrote %s and %s\n", csv_path, summary_path);
    return 0;
}
