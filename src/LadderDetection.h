#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include "GameState.h"
#include "GTP.h"

int IsLadderEscape(
    const GameState* const state,
    const int &str_vtx,
    const bool &chase = false
);

int IsLadderChase(
    const GameState* const state,
    const int &chase_vtx,
    const GameState* base_state
);

void LadderDetection(
    const GameState* state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes = INT_MAX
);
#endif
