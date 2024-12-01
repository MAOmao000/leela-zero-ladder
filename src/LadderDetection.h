#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include "GameState.h"

void LadderDetection(
    const GameState* const state,
    int *ladder_pos,
    const std::array<float, NUM_INTERSECTIONS>& policy
);
#endif
