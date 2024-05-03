#ifndef _LADDER_DETECTION_H_
#define _LADDER_DETECTION_H_

//#include "GoBoard.h"
#include "GameState.h"

#define LADDER      1
#define LADDER_LIKE 2

// 
void LadderDetection(const GameState &state, char *ladder_pos, bool is_root=true);
// 
#endif
