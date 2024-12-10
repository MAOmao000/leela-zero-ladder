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

static bool IsLadderCaptured(
    int &depth,
    std::unique_ptr<GameState> &state,
    const int str_vtx,
    const int turn_color,
    const int max_ladder_depth,
    const bool escape = true,
    int escape_pos = 0)
{
    auto escape_color = state->board.get_state(str_vtx);
    int num_liberty;
    if (escape_pos && escape_color == FastBoard::EMPTY) {
        num_liberty = 1;
    } else {
        if (escape_color == FastBoard::EMPTY) {
            if (state->m_komove != FastBoard::NO_VERTEX) {
                return ALIVE;
            } else {
                return DEAD;
            }
        }
        num_liberty = state->board.get_liberties(str_vtx);
    }
    if (depth >= max_ladder_depth) {
        if (escape) {
            return DEAD;
        } else {
            return ALIVE;
        }
    }
    auto base_depth = depth;
    if (turn_color == escape_color || escape_color == FastBoard::EMPTY) {
        auto max_depth_alive = 0;
        auto min_depth_dead = max_ladder_depth + 1;
        if (turn_color == escape_color) {
            if (state->m_komove != FastBoard::NO_VERTEX || num_liberty >= 2) {
                return ALIVE;
            }
            // Check if can capture the stone of the surrounding opponent.
            char capture_checked[FastBoard::NUM_VERTICES] = {};
            auto newpos = str_vtx;
            auto n_vtx = 0;
            do {
                // Check whether can capture the stone at the breathing point of opponent's stones.
                for (auto d = 0; d < 4; d++) {
                    n_vtx = state->board.get_state_neighbor(newpos, d);
                    if (state->board.get_state(n_vtx) != FLIP_COLOR(escape_color) ||
                        state->board.get_liberties(n_vtx) != 1) {
                        continue;
                    }
                    if (capture_checked[state->board.get_parent_stone(n_vtx)]) {
                        continue;
                    }
                    capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                    auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                    if (state->is_move_legal(turn_color, liberty_pos[0])) {
                        state->play_move(turn_color, liberty_pos[0]);
                        depth = base_depth;
                        if (IsLadderCaptured(
                                ++depth,
                                state,
                                str_vtx,
                                FLIP_COLOR(turn_color),
                                max_ladder_depth,
                                escape
                            ) == ALIVE) {
                            if (escape) {
                                state->undo_move();
                                return ALIVE;
                            }
                            if (depth > max_depth_alive) {
                                max_depth_alive = depth;
                            }
                        } else {
                            if (depth < min_depth_dead) {
                                min_depth_dead = depth;
                            }
                        }
                        state->undo_move();
                    }
                }
                newpos = state->board.get_next_stone(newpos);
            } while (newpos != str_vtx);
        }

        if (!escape_pos) {
            auto liberty_pos = state->board.get_liberty_pos(1, str_vtx);
            escape_pos = liberty_pos[0];
        }
        if (state->is_move_legal(turn_color, escape_pos)) {
            state->play_move(turn_color, escape_pos);
            depth = base_depth;
            if (IsLadderCaptured(
                    ++depth,
                    state,
                    str_vtx,
                    FLIP_COLOR(turn_color),
                    max_ladder_depth,
                    escape
                ) == ALIVE) {
                state->undo_move();
                if (depth < max_depth_alive) {
                    depth = max_depth_alive;
                }
                return ALIVE;
            } else {
                state->undo_move();
                if (max_depth_alive) {
                    depth = max_depth_alive;
                    return ALIVE;
                } else if (depth > min_depth_dead) {
                    depth = min_depth_dead;
                    return DEAD;
                }
                return DEAD;
            }
        }
        if (max_depth_alive) {
            depth = max_depth_alive;
            return ALIVE;
        } else if (min_depth_dead <= max_ladder_depth) {
            depth = min_depth_dead;
        }
        return DEAD;
    } else {
        if (state->m_komove != FastBoard::NO_VERTEX || num_liberty >= 3) {
             return ALIVE;
        }
        auto max_depth_alive = 0;
        auto min_depth_dead = max_ladder_depth + 1;
        auto liberty_pos = state->board.get_liberty_pos(2, str_vtx);
        for (auto i = 0; i < 2; i++) {
            if (liberty_pos[i] && state->is_move_legal(turn_color, liberty_pos[i])) {
                state->play_move(turn_color, liberty_pos[i]);
                depth = base_depth;
                if (IsLadderCaptured(
                        ++depth,
                        state,
                        str_vtx,
                        FLIP_COLOR(turn_color),
                        max_ladder_depth,
                        escape
                    ) == DEAD) {
                    if (!escape) {
                        state->undo_move();
                        return DEAD;
                    }
                    min_depth_dead = std::min(depth, min_depth_dead);
                } else {
                    max_depth_alive = std::max(depth, max_depth_alive);
                }
                state->undo_move();
            }
        }
        if (min_depth_dead <= max_ladder_depth) {
            depth = min_depth_dead;
            return DEAD;
        } else if (max_depth_alive) {
            depth = max_depth_alive;
        }
        return ALIVE;
    }
    return ALIVE;
}

void LadderDetection(
    const GameState* const state,
    int *ladder_pos,
    const std::array<float, NUM_INTERSECTIONS>& policy)
{
    auto state_copy = std::make_unique<GameState>(state);
    const auto turn_color = state_copy->board.get_to_move();
    const auto opponent = FLIP_COLOR(turn_color);

    auto depth = 0;
    char ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state_copy->board.get_vertex(x, y);

        if (cfg_defense_stones < 1 &&
            state_copy->board.get_state(vertex) == FastBoard::EMPTY &&
            (policy[0] < 0.0f || policy[i] > cfg_ladder_min_policy_defense)) {
            auto liberty_count = 0;
            for (auto d = 0; d < 4; d++) {
                auto n_vtx = state_copy->board.get_state_neighbor(vertex, d);
                if (state_copy->board.get_state(n_vtx) == turn_color) {
                    liberty_count = 0;
                    break;
                } else if (state_copy->board.get_state(n_vtx) == FastBoard::EMPTY) {
                    liberty_count++;
                }
            }
            if (liberty_count == 2) {
                depth = 0;
                if (IsLadderCaptured(
                        depth,
                        state_copy,
                        vertex,
                        turn_color,
                        cfg_ladder_depth,
                        true,
                        vertex
                    ) == DEAD) {
                    ladder_pos[i] = depth;
                }
            }
        } else if (state_copy->board.get_state(vertex) == turn_color &&
            !ladder_checked[state_copy->board.get_parent_stone(vertex)] &&
            state_copy->board.get_string_count(vertex) >= cfg_defense_stones &&
            state_copy->board.get_liberties(vertex) == 1) {
            ladder_checked[state_copy->board.get_parent_stone(vertex)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, vertex);

            auto xy = state_copy->board.get_xy(liberty_pos[0]);
            if (policy[0] < 0.0f ||
                policy[xy.first + xy.second * BOARD_SIZE] > cfg_ladder_min_policy_defense) {
                depth = 0;
                if (IsLadderCaptured(
                        depth,
                        state_copy,
                        vertex,
                        turn_color,
                        cfg_ladder_depth,
                        true,
                        liberty_pos[0]
                    ) == DEAD) {
                    if (ladder_pos[xy.first + xy.second * BOARD_SIZE] <= 0) {
                        if (-1 * ladder_pos[xy.first + xy.second * BOARD_SIZE] < depth) {
                            ladder_pos[xy.first + xy.second * BOARD_SIZE] = depth;
                        }
                    } else if (ladder_pos[xy.first + xy.second * BOARD_SIZE] < depth) {
                        ladder_pos[xy.first + xy.second * BOARD_SIZE] = depth;
                    }
                }
            }
        } else if (state_copy->board.get_state(vertex) == opponent &&
                   !ladder_checked[state_copy->board.get_parent_stone(vertex)] &&
                   state_copy->board.get_string_count(vertex) >= cfg_offense_stones &&
                   state_copy->board.get_liberties(vertex) == 2) {
            ladder_checked[state_copy->board.get_parent_stone(vertex)] = CHECKED;
            // Check the opponent's stone with two breathing points.
            auto liberty_pos = state->board.get_liberty_pos(2, vertex);
            // Checking the stone of the current turn with two breathing points.
            for (auto i = 0; i < 2; i++) {
                auto xy = state_copy->board.get_xy(liberty_pos[i]);
                auto ladder_idx = xy.first + xy.second * BOARD_SIZE;
                if (policy[0] < 0.0f || policy[ladder_idx] > cfg_ladder_min_policy_offense) {
                    if (state_copy->is_move_legal(turn_color, liberty_pos[i])) {
                        state_copy->play_move(turn_color, liberty_pos[i]);
                        depth = 0;
                        if (IsLadderCaptured(
                                depth,
                                state_copy,
                                vertex,
                                opponent,
                                cfg_ladder_depth,
                                false
                            ) == ALIVE) {
                            if (ladder_pos[ladder_idx] <= 0) {
                                if (-1 * ladder_pos[ladder_idx] < depth) {
                                    ladder_pos[ladder_idx] = -1 * depth;
                                }
                            } else if (ladder_pos[ladder_idx] < depth) {
                                ladder_pos[ladder_idx] = -1 * depth;
                            }
                        } else {
                            if (ladder_pos[ladder_idx] <= 0) {
                                ladder_pos[ladder_idx] = depth + cfg_ladder_depth + 1;
                            }
                        }
                        state_copy->undo_move();
                    }
                }
            }
        }
    }
}
