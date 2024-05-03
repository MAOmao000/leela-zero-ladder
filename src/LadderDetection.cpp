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

static bool IsLadderCaptured(int &depth, std::unique_ptr<GameState> &state, const int str_vtx, const int turn_color);

////////////////////////////////
//                            //
////////////////////////////////
void LadderDetection(const GameState &state, char *ladder_pos, bool is_root)
{
    if (cfg_root_ladder && !is_root) return;

    auto state_copy = std::make_unique<GameState>(state);
    const auto turn_color = state_copy->board.get_to_move();
    const auto opponent = turn_color ^ 0x01;

    if (state_copy->m_komove != FastBoard::NO_VERTEX) return;

    char ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state_copy->board.get_vertex(x, y);

        if (state_copy->board.get_state(vertex) == turn_color &&
            state_copy->board.get_liberties(vertex) == 1) {
            // Checking the stone of the current turn with one breathing point.
            auto newpos = vertex;
            auto liberty_pos = 0;
            // Follow the connecting stones and find the breathing point.
            do {
                if (ladder_checked[newpos]) {
                    break;
                }
                if (!liberty_pos) {
                    for (auto d = 0; d < 4; d++) {
                        // Check if the target has one liberty in the opponent's stone.
                        auto n_vtx = state_copy->board.get_state_neighbor(newpos, d);
                        if (state_copy->board.get_state(n_vtx) == FastBoard::EMPTY) {
                            liberty_pos = n_vtx;
                            // Check whether the target stone is a stone that cannot be escaped.
                            if (state_copy->is_move_legal(turn_color, liberty_pos)) {
                                state_copy->play_move(turn_color, liberty_pos);
                                auto depth = 0;
                                if (IsLadderCaptured(depth, state_copy, vertex, opponent) == DEAD) {
                                    if (depth >= cfg_ladder_defense) {
                                        ladder_pos[liberty_pos] = LADDER;
                                    }
                                }
                                state_copy->undo_move();
                            }
                            break;
                        }
                    }
                }
                ladder_checked[newpos] = CHECKED;
                newpos = state_copy->board.get_next_stone(newpos);
            } while (newpos != vertex);
        } else if ((!cfg_root_offense || (cfg_root_offense && is_root)) &&
                   state_copy->board.get_state(vertex) == opponent &&
                   state_copy->board.get_string_count(vertex) >= cfg_offense_stones &&
                   state_copy->board.get_liberties(vertex) == 2) {
            // Check the opponent's stone with two breathing points.
            auto newpos = vertex;
            auto liberty_cnt = 0;
            auto liberty_pos = 0;
            do {
                if (liberty_cnt < 2 && !ladder_checked[newpos]) {
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state_copy->board.get_state_neighbor(newpos, d);
                        if (n_vtx == liberty_pos) continue;
                        if (state_copy->board.get_state(n_vtx) == FastBoard::EMPTY) {
                            // If it's empty, move a stone
                            liberty_cnt++;
                            liberty_pos = n_vtx;
                            if (state_copy->is_move_legal(turn_color, n_vtx) && !ladder_pos[n_vtx]) {
                                state_copy->play_move(turn_color, n_vtx);
                                if (state_copy->board.get_next_stone(n_vtx) == n_vtx) {
                                    auto depth = 0;
                                    if (IsLadderCaptured(depth, state_copy, vertex, opponent) == ALIVE) {
                                        if (depth >= cfg_ladder_offense) {
                                            ladder_pos[n_vtx] = LADDER_LIKE;
                                        }
                                    } else {
                                        if (ladder_pos[n_vtx] == LADDER_LIKE) ladder_pos[n_vtx] = 0;
                                        break;
                                    }
                                }
                                state_copy->undo_move();
                            }
                            if (liberty_cnt >= 2) break;
                        }
                    }
                }
                ladder_checked[newpos] = CHECKED;
                newpos = state_copy->board.get_next_stone(newpos);
            } while (newpos != vertex);
        }
    }
}
////////////////////
//                //
////////////////////
static bool IsLadderCaptured(int &depth, std::unique_ptr<GameState> &state, const int str_vtx, const int turn_color)
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
    } else if (num_liberty >= 3) {
        return ALIVE;
    }

    auto capture_color = escape_color ^ 0x01;
    auto base_depth = depth;
    auto max_depth_alive = depth;

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
                if (capture_checked[n_vtx]) continue;
                capture_checked[n_vtx] = CHECKED;
                if (state->board.get_state(n_vtx) != capture_color ||
                    state->board.get_liberties(n_vtx) != 1) continue;
                auto liberty_pos = n_vtx;
                // Find the breathing point of opponent's stones.
                do {
                    for (auto l = 0; l < 4; l++) {
                        auto l_vtx = state->board.get_state_neighbor(liberty_pos, l);
                        if (state->board.get_state(l_vtx) == FastBoard::EMPTY) {
                            if (state->is_move_legal(escape_color, l_vtx)) {
                                state->play_move(escape_color, l_vtx);
                                depth = base_depth;
                                if (IsLadderCaptured(++depth, state, str_vtx, turn_color ^ 0x01) == ALIVE) {
                                    state->undo_move();
                                    return ALIVE;
                                }
                                state->undo_move();
                            }
                            newpos = 0;
                            break;
                        }
                    }
                    if (!newpos) break;
                    liberty_pos = state->board.get_next_stone(liberty_pos);
                } while (liberty_pos != n_vtx);
                if (!newpos) break;
            }
            if (!newpos) break;
            newpos = state->board.get_next_stone(newpos);
        } while (newpos != str_vtx);

        char escape_checked[FastBoard::NUM_VERTICES] = {};
        newpos = str_vtx;
        do {
            for (auto d = 0; d < 4; d++) {
                n_vtx = state->board.get_state_neighbor(newpos, d);
                if (escape_checked[n_vtx]) continue;
                escape_checked[n_vtx] = CHECKED;
                if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                    if (state->is_move_legal(escape_color, n_vtx)) {
                        state->play_move(escape_color, n_vtx);
                        depth = base_depth;
                        if (IsLadderCaptured(++depth, state, str_vtx, turn_color ^ 0x01) == ALIVE) {
                            state->undo_move();
                            return ALIVE;
                        } else {
                            state->undo_move();
                            return DEAD;
                        }
                        state->undo_move();
                    }
                    return DEAD;
                }
            }
            newpos = state->board.get_next_stone(newpos);
        } while (newpos != str_vtx);
        return DEAD;
    } else {
        if (num_liberty == 1) return DEAD;
        char escape_checked[FastBoard::NUM_VERTICES] = {};
        auto newpos = str_vtx;
        auto n_vtx = 0;
        auto liberty_cnt = 0;
        do {
            for (auto d = 0; d < 4; d++) {
                n_vtx = state->board.get_state_neighbor(newpos, d);
                if (escape_checked[n_vtx]) continue;
                escape_checked[n_vtx] = CHECKED;
                if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                    liberty_cnt++;
                    if (state->is_move_legal(capture_color, n_vtx)) {
                        state->play_move(capture_color, n_vtx);
                        depth = base_depth;
                            if (IsLadderCaptured(++depth, state, str_vtx, turn_color ^ 0x01) == DEAD) {
                            state->undo_move();
                            return DEAD;
                        }
                        state->undo_move();
                        if (depth > max_depth_alive) max_depth_alive = depth;
                    }
                    if (liberty_cnt >= num_liberty) {
                        depth = max_depth_alive;
                        return ALIVE;
                    }
                }
            }
            newpos = state->board.get_next_stone(newpos);
        } while (newpos != str_vtx);
    }
    depth = max_depth_alive;
    return ALIVE;
}
