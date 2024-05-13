#include <iostream>
#include <memory>

#include "LadderDetection.h"
#include "GTP.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

#define ALIVE  1
#define DEAD   0
#define CHECKED 1
#define FLIP_COLOR(col) ((col) ^ 0x01)

static bool IsLadderCaptured(int &depth, std::unique_ptr<GameState> &state, const int str_vtx, const int turn_color, int escape_pos = 0);

////////////////////////////////
//                            //
////////////////////////////////
void LadderDetection(const GameState* const state, char *ladder_pos)
{
    auto state_copy = std::make_unique<GameState>(state);
    const auto turn_color = state_copy->board.get_to_move();
    const auto opponent = FLIP_COLOR(turn_color);

    if (state_copy->m_komove != FastBoard::NO_VERTEX) return;

    auto depth = 0;
    char ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state_copy->board.get_vertex(x, y);

        if (state_copy->board.get_state(vertex) == turn_color &&
            state_copy->board.get_string_count(vertex) >= cfg_defense_stones &&
            state_copy->board.get_liberties(vertex) == 1) {
            auto newpos = vertex;
            auto liberty_pos = 0;
            // Follow the connecting stones and find the breathing point.
            do {
                if (ladder_checked[newpos]) break;
                if (!liberty_pos) {
                    for (auto d = 0; d < 4; d++) {
                        // Check if the target has one liberty in the opponent's stone.
                        auto n_vtx = state_copy->board.get_state_neighbor(newpos, d);
                        if (state_copy->board.get_state(n_vtx) == FastBoard::EMPTY) {
                            liberty_pos = n_vtx;
                            break;
                        }
                    }
                }
                ladder_checked[newpos] = CHECKED;
                newpos = state_copy->board.get_next_stone(newpos);
            } while (newpos != vertex);
            if (liberty_pos) {
                // Checking the stone of the current turn with one breathing point.
                depth = 0;
                if (IsLadderCaptured(depth, state_copy, vertex, turn_color, liberty_pos) == DEAD) {
                    if (depth >= cfg_ladder_defense) {
                        auto xy = state_copy->board.get_xy(liberty_pos);
                        ladder_pos[xy.first + xy.second * BOARD_SIZE] = LADDER;
                    }
                }
            }
        } else if (state_copy->board.get_state(vertex) == opponent &&
                   state_copy->board.get_string_count(vertex) >= cfg_offense_stones &&
                   state_copy->board.get_liberties(vertex) == 2) {
            // Check the opponent's stone with two breathing points.
            auto newpos = vertex;
            auto liberty_cnt = 0;
            std::array<int, 2> liberty_pos = {0, 0};
            do {
                if (liberty_cnt < 2 && !ladder_checked[newpos]) {
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state_copy->board.get_state_neighbor(newpos, d);
                        if (state_copy->board.get_state(n_vtx) == FastBoard::EMPTY &&
                            n_vtx != liberty_pos[0]) {
                            liberty_pos[liberty_cnt] = n_vtx;
                            liberty_cnt++;
                            if (liberty_cnt >= 2) break;
                        }
                    }
                }
                ladder_checked[newpos] = CHECKED;
                newpos = state_copy->board.get_next_stone(newpos);
            } while (newpos != vertex);
            // Checking the stone of the current turn with two breathing points.
            if (liberty_pos[0]) {
                auto xy = state_copy->board.get_xy(liberty_pos[0]);
                auto ladder_idx = xy.first + xy.second * BOARD_SIZE;
                if (state_copy->is_move_legal(turn_color, liberty_pos[0]) &&
                    !ladder_pos[ladder_idx]) {
                    state_copy->play_move(turn_color, liberty_pos[0]);
                    if (state_copy->board.get_next_stone(liberty_pos[0]) == liberty_pos[0]) {
                        depth = 0;
                        if (IsLadderCaptured(depth, state_copy, vertex, opponent) == ALIVE) {
                            if (depth >= cfg_ladder_offense) {
                                ladder_pos[ladder_idx] = LADDER_LIKE;
                            }
                        } else {
                            if (ladder_pos[ladder_idx] == LADDER_LIKE) {
                                ladder_pos[ladder_idx] = 0;
                            }
                        }
                    }
                    state_copy->undo_move();
                }
            }
            if (liberty_pos[1]) {
                auto xy = state_copy->board.get_xy(liberty_pos[1]);
                auto ladder_idx = xy.first + xy.second * BOARD_SIZE;
                if (state_copy->is_move_legal(turn_color, liberty_pos[1]) &&
                    !ladder_pos[ladder_idx]) {
                    state_copy->play_move(turn_color, liberty_pos[1]);
                    if (state_copy->board.get_next_stone(liberty_pos[1]) == liberty_pos[1]) {
                        depth = 0;
                        if (IsLadderCaptured(depth, state_copy, vertex, opponent) == ALIVE) {
                            if (depth >= cfg_ladder_offense) {
                                ladder_pos[ladder_idx] = LADDER_LIKE;
                            }
                        } else {
                            if (ladder_pos[ladder_idx] == LADDER_LIKE) {
                                ladder_pos[ladder_idx] = 0;
                            }
                        }
                    }
                    state_copy->undo_move();
                }
            }
        }
    }
}
////////////////////
//                //
////////////////////
static bool IsLadderCaptured(int &depth, std::unique_ptr<GameState> &state, const int str_vtx, const int turn_color, int escape_pos)
{
    if (state->m_komove != FastBoard::NO_VERTEX) {
        return ALIVE;
    } else if (depth >= cfg_ladder_depth) {
        return DEAD;
    }

    auto escape_color = state->board.get_state(str_vtx);
    auto num_liberty = state->board.get_liberties(str_vtx);
    if (escape_color == FastBoard::EMPTY) {
        return DEAD;
    }

    auto capture_color = FLIP_COLOR(escape_color);
    auto base_depth = depth;
    auto max_depth_alive = 0;
    auto max_depth_dead = 0;

    if (turn_color == escape_color) {
        if (num_liberty >= 2) return ALIVE;
        char capture_checked[FastBoard::NUM_VERTICES] = {};
        auto newpos = str_vtx;
        auto n_vtx = 0;
        // Check if can capture the stone of the surrounding opponent.
        do {
            // Check whether can capture the stone at the breathing point of opponent's stones.
            for (auto d = 0; d < 4; d++) {
                n_vtx = state->board.get_state_neighbor(newpos, d);
                if (capture_checked[state->board.get_parent_stone(n_vtx)]) continue;
                capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                if (state->board.get_state(n_vtx) != capture_color ||
                    state->board.get_liberties(n_vtx) != 1) continue;
                auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                if (state->is_move_legal(turn_color, liberty_pos[0])) {
                    state->play_move(turn_color, liberty_pos[0]);
                    depth = base_depth;
                    if (IsLadderCaptured(++depth, state, str_vtx, FLIP_COLOR(turn_color)) == ALIVE) {
                        state->undo_move();
                        return ALIVE;
                    }
                    state->undo_move();
                }
            }
            newpos = state->board.get_next_stone(newpos);
        } while (newpos != str_vtx);
        if (!escape_pos) {
            auto liberty_pos = state->board.get_liberty_pos(1, str_vtx);
            escape_pos = liberty_pos[0];
        }
        if (state->is_move_legal(turn_color, escape_pos)) {
            state->play_move(turn_color, escape_pos);
            depth = base_depth;
            if (IsLadderCaptured(++depth, state, str_vtx, FLIP_COLOR(turn_color)) == ALIVE) {
                state->undo_move();
                return ALIVE;
            } else {
                state->undo_move();
                return DEAD;
            }
        }
        return DEAD;
    } else {
        if (num_liberty >= 3) return ALIVE;
        auto liberty_pos = state->board.get_liberty_pos(2, str_vtx);
        if (liberty_pos[0]) {
            if (state->is_move_legal(turn_color, liberty_pos[0])) {
                state->play_move(turn_color, liberty_pos[0]);
                depth = base_depth;
                if (IsLadderCaptured(++depth, state, str_vtx, FLIP_COLOR(turn_color)) == DEAD) {
                    max_depth_dead = depth;
                } else {
                    max_depth_alive = depth;
                }
                state->undo_move();
            }
        }
        if (liberty_pos[1]) {
            if (state->is_move_legal(turn_color, liberty_pos[1])) {
                state->play_move(turn_color, liberty_pos[1]);
                depth = base_depth;
                if (IsLadderCaptured(++depth, state, str_vtx, FLIP_COLOR(turn_color)) == DEAD) {
                    state->undo_move();
                    if (depth < max_depth_dead) depth = max_depth_dead;
                    return DEAD;
                } else {
                    state->undo_move();
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
    if (max_depth_dead) {
        depth = max_depth_dead;
        return DEAD;
    }
    depth = max_depth_alive;
    return ALIVE;
}
