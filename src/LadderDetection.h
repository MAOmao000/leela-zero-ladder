#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

#include <climits>

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
    Network::Netresult& result
);
#endif
