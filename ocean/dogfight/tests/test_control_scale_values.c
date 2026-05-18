/*
 * test_control_scale_values.c - 1:1 port of test_high_speed_oscillation.py
 *   ::test_control_scale_values (the formula-only sub-test).
 *
 * The simulation half of test_high_speed_oscillation.py is already covered by
 * ocean/dogfight/test_flight_dynamics.c and
 * ocean/dogfight/tests/test_flight_physics.c::test_high_speed_pitch_oscillation.
 * Do NOT re-port that here.
 *
 * Dogfight 3.0 trained with full control authority at all speeds. Keep this
 * test hard-failing so later smoothness tuning does not silently reduce the
 * agent's actuator authority again.
 */
#include <stdio.h>
#include <math.h>

#include "../flightlib.h"

typedef struct {
    int speed;
    float expected;  // dogfight3 expected scale (no high-speed reduction)
} Case;

static const Case CASES[] = {
    { 80, 1.00f},
    {100, 1.00f},
    {120, 1.00f},
    {140, 1.00f},
    {160, 1.00f},
    {180, 1.00f},
    {200, 1.00f},
    { 60, 1.00f},
};
#define N_CASES (sizeof(CASES) / sizeof(CASES[0]))

static float control_scale(float speed) {
    /* Mirrors the C formula in flightlib.h compute_control_scale(). */
    float over = speed - CONTROL_V_REF;
    if (over < 0.0f) over = 0.0f;
    float scale = 1.0f - over * CONTROL_SCALE_SLOPE;
    if (scale < CONTROL_SCALE_MIN) scale = CONTROL_SCALE_MIN;
    return scale;
}

int main(void) {
    printf("\nVerifying control scale formula against dogfight3...\n");
    printf("--------------------------------------------------------\n");
    printf("flightlib.h: V_REF=%.1f SLOPE=%.4f MIN=%.2f\n",
           CONTROL_V_REF, CONTROL_SCALE_SLOPE, CONTROL_SCALE_MIN);

    int n_ok = 0;
    int n_diff = 0;
    for (size_t i = 0; i < N_CASES; ++i) {
        float scale = control_scale((float)CASES[i].speed);
        int match = fabsf(scale - CASES[i].expected) < 0.001f;
        const char* status = match ? "OK" : "FAIL";
        if (match) ++n_ok; else ++n_diff;
        printf("V=%3d m/s: scale=%.2f (dogfight3 expected %.2f) [%s]\n",
               CASES[i].speed, scale, CASES[i].expected, status);
    }

    printf("--------------------------------------------------------\n");
    printf("Matches dogfight3: %d/%zu\n", n_ok, N_CASES);

    if (fabsf(default_flight_params().damping_multiplier - 1.0f) >= 0.001f) {
        printf("damping_multiplier=%.2f (dogfight3 expected 1.00) [FAIL]\n",
               default_flight_params().damping_multiplier);
        ++n_diff;
    }

    return n_diff;
}
