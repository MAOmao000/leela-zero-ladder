#include <iostream>
#include <memory>

#include "Ladder.h"
#include "SearchBoard.h"
#include "GTP.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

#define ALIVE  1
#define DEAD   0

//
static bool IsLadderCaptured(int &depth, search_game_info_t *game, const int ren_xy, const int turn_color);

////////////////////////////////
//                            //
////////////////////////////////
void LadderExtension(const GameState* const state, char *ladder_pos)
{
    if (state->m_komove != FastBoard::NO_VERTEX) {
        return;
    }

    auto color = state->board.black_to_move() ? S_BLACK : S_WHITE;
    game_info_t *game = AllocateGame();
    InitializeBoard(game);
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            auto vertex = state->board.get_vertex(col, row);
            auto stone = state->board.get_state(vertex);
            if (stone < 2) { // stone = 0: BLACK, stone = 1: WHITE
                PutStone(game, POS(col + BOARD_START, row + BOARD_START), stone ? S_WHITE : S_BLACK);
            }
        }
    }
    const string_t *string = game->string;
    std::unique_ptr<search_game_info_t> search_game = nullptr;
    auto depth = 0;
    int neighbor4[4];

    for (int i = 0; i < MAX_STRING; i++) {
        if (!string[i].flag) continue;
        if (cfg_ladder_defense > 0 &&
            string[i].color == color &&
            string[i].size >= cfg_defense_stones &&
            string[i].libs == 1) {
            char flag = DEAD;
            if (!search_game)
                search_game.reset(new search_game_info_t(game));
            search_game_info_t *ladder_game = search_game.get();
            int neighbor = string[i].neighbor[0]; // neighbor: opponent's stone
            while (neighbor != NEIGHBOR_END && flag == DEAD) {
                if (string[neighbor].libs == 1) {
                    if (IsLegalForSearch(ladder_game, string[neighbor].lib[0], color)) {
                        PutStoneForSearch(ladder_game, string[neighbor].lib[0], color);
                        depth = 0;
                        if (IsLadderCaptured(depth, ladder_game, string[i].origin, FLIP_COLOR(color) == ALIVE) ||
                            depth < cfg_ladder_defense) {
                            flag = ALIVE;
                        }
                        Undo(ladder_game);
                    }
                    break;
                }
                neighbor = string[i].neighbor[neighbor];
            }
            if (flag == DEAD) {
                if (IsLegalForSearch(ladder_game, string[i].lib[0], color)) {
                    PutStoneForSearch(ladder_game, string[i].lib[0], color);
                    depth = 0;
                    if (IsLadderCaptured(depth, ladder_game, string[i].origin, FLIP_COLOR(color)) == DEAD) {
                        if (depth >= cfg_ladder_defense) {
                            auto x = CORRECT_X(string[i].lib[0]) - 1;
                            auto y = CORRECT_Y(string[i].lib[0]) - 1;
                            ladder_pos[x + y * BOARD_SIZE] = LADDER;
                        }
                    }
                    Undo(ladder_game);
                }
            }
        } else if (cfg_ladder_offense > 0 &&
                   string[i].color == FLIP_COLOR(color) &&
                   string[i].size >= cfg_offense_stones &&
                   string[i].libs == 2) {
            if (!search_game)
                search_game.reset(new search_game_info_t(game));
            depth = 0;
            search_game_info_t *ladder_game = search_game.get();
            if (IsLegalForSearch(ladder_game, string[i].lib[0], color)) {
                PutStoneForSearch(ladder_game, string[i].lib[0], color);
                GetNeighbor4(neighbor4, string[i].lib[0]);
                if (ladder_game->board[neighbor4[0]] != color &&
                    ladder_game->board[neighbor4[1]] != color &&
                    ladder_game->board[neighbor4[2]] != color &&
                    ladder_game->board[neighbor4[3]] != color) {
                    depth = 0;
                    if (IsLadderCaptured(depth, ladder_game, string[i].origin, FLIP_COLOR(color)) == ALIVE) {
                        if (depth >= cfg_ladder_offense) {
                            auto x = CORRECT_X(string[i].lib[0]) - 1;
                            auto y = CORRECT_Y(string[i].lib[0]) - 1;
                            ladder_pos[x + y * BOARD_SIZE] = LADDER_LIKE;
                        }
                    } else {
                        auto x = CORRECT_X(string[i].lib[0]) - 1;
                        auto y = CORRECT_Y(string[i].lib[0]) - 1;
                        if (ladder_pos[x + y * BOARD_SIZE] == LADDER_LIKE) {
                            ladder_pos[x + y * BOARD_SIZE] = 0;
                        }
                    }
                }
                Undo(ladder_game);
            }
            auto second_lib = string[i].lib[string[i].lib[0]];
            if (IsLegalForSearch(ladder_game, second_lib, color)) {
                PutStoneForSearch(ladder_game, second_lib, color);
                GetNeighbor4(neighbor4, second_lib);
                if (ladder_game->board[neighbor4[0]] != color &&
                    ladder_game->board[neighbor4[1]] != color &&
                    ladder_game->board[neighbor4[2]] != color &&
                    ladder_game->board[neighbor4[3]] != color) {
                    depth = 0;
                    if (IsLadderCaptured(depth, ladder_game, string[i].origin, FLIP_COLOR(color)) == ALIVE) {
                        if (depth >= cfg_ladder_offense) {
                            auto x = CORRECT_X(second_lib) - 1;
                            auto y = CORRECT_Y(second_lib) - 1;
                            ladder_pos[x + y * BOARD_SIZE] = LADDER_LIKE;
                        }
                    } else {
                        auto x = CORRECT_X(second_lib) - 1;
                        auto y = CORRECT_Y(second_lib) - 1;
                        if (ladder_pos[x + y * BOARD_SIZE] == LADDER_LIKE) {
                            ladder_pos[x + y * BOARD_SIZE] = 0;
                        }
                    }
                }
                Undo(ladder_game);
            }
        }
    }
    FreeGame(game);
}
////////////////////
//                //
////////////////////
static bool IsLadderCaptured(int &depth, search_game_info_t *game, const int ren_xy, const int turn_color)
{
    const char *board = game->board;
    const string_t *string = game->string;
    const int str = game->string_id[ren_xy];
    int escape_color, capture_color;
    int escape_xy, capture_xy;
    int neighbor; // , base_depth , max_depth;
    bool result;

    if (game->ko_move == (game->moves - 1)) {
        return ALIVE;
    } else if (depth >= cfg_ladder_depth) {
        return DEAD;
    }

    if (board[ren_xy] == S_EMPTY) {
        return DEAD;
    }

    escape_color = board[ren_xy];
    capture_color = FLIP_COLOR(escape_color);
    auto base_depth = depth;
    auto max_depth_alive = 0;
    auto max_depth_dead = 0;

    if (turn_color == escape_color) {
        if (string[str].libs >= 2) return ALIVE;
        neighbor = string[str].neighbor[0];
        while (neighbor != NEIGHBOR_END) {
            if (string[neighbor].libs == 1) {
                if (IsLegalForSearch(game, string[neighbor].lib[0], escape_color)) {
                    PutStoneForSearch(game, string[neighbor].lib[0], escape_color);
                    depth = base_depth;
                    result = IsLadderCaptured(++depth, game, ren_xy, FLIP_COLOR(turn_color));
                    Undo(game);
                    if (result == ALIVE) {
                        return ALIVE;
                    }
                }
            }
            neighbor = string[str].neighbor[neighbor];
        }
        escape_xy = string[str].lib[0];
        if (IsLegalForSearch(game, escape_xy, escape_color)) {
            PutStoneForSearch(game, escape_xy, escape_color);
            depth = base_depth;
            result = IsLadderCaptured(++depth, game, ren_xy, FLIP_COLOR(turn_color));
            Undo(game);
            return result;
        }
        return DEAD;
    } else {
        if (string[str].libs >= 3) {
            return ALIVE;
        }
        capture_xy = string[str].lib[0];
        if (capture_xy != LIBERTY_END) {
            if (IsLegalForSearch(game, capture_xy, capture_color)) {
                PutStoneForSearch(game, capture_xy, capture_color);
                depth = base_depth;
                result = IsLadderCaptured(++depth, game, ren_xy, FLIP_COLOR(turn_color));
                Undo(game);
                if (result == DEAD) {
                    max_depth_dead = depth;
                } else {
                    max_depth_alive = depth;
                }
            }
            capture_xy = string[str].lib[capture_xy];
            if (capture_xy != LIBERTY_END) {
                if (IsLegalForSearch(game, capture_xy, capture_color)) {
                    PutStoneForSearch(game, capture_xy, capture_color);
                    depth = base_depth;
                    result = IsLadderCaptured(++depth, game, ren_xy, FLIP_COLOR(turn_color));
                    Undo(game);
                    if (result == DEAD) {
                        if (depth < max_depth_dead) depth = max_depth_dead;
                        return DEAD;
                    } else {
                        if (max_depth_dead) {
                            depth = max_depth_dead;
                            return DEAD;
                        } else if (depth < max_depth_alive) {
                            depth = max_depth_alive;
                        }
                        return ALIVE;
                    }
                }
            }
        }
    }
    if (max_depth_dead) {
        depth = max_depth_dead;
        return DEAD;
    }
    if (depth < max_depth_alive) depth = max_depth_alive;
    return ALIVE;
}
