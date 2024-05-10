#ifndef _LADDER_H_
#define _LADDER_H_

#include "GameState.h"
#include "GoBoard.h"

#define LADDER      1
#define LADDER_LIKE 2

// 
void LadderExtension(const GameState* const state, char *ladder_pos);
//void LadderExtension( game_info_t *game, int color, char *ladder_pos );
// 
#endif
