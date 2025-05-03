#include <iostream>
#include <memory>
#include <cmath>

#include "LadderDetection.h"
#include "GTP.h"
#include "Utils.h"

using namespace std;
using namespace Utils;

#define ALIVE false
#define DEAD true
#define CHECKED 1
#define FLIP_COLOR(col) ((col) ^ 0x01)

static constexpr int MAX_LADDER_SEARCH_NODE_BUDGET = 25000;
static constexpr int stackSize = 100;

int IsLadderEscape(
    const GameState* const base_state,
    const int &str_vtx,
    const bool &chase
    )
{
    auto state = std::make_unique<GameState>(base_state);
    int moveListStarts[stackSize];
    int moveListLens[stackSize];
    int moveListCur[stackSize];
    int moveListAlive[stackSize];
    std::array<int, stackSize * 2> buf;

    int max_depth = 0;
    int dead_depth = 0;
    bool returnValue = ALIVE;

    int stackIdx = 0;
    int searchNodeCount = 0;
    moveListCur[0] = -1;
    moveListStarts[0] = 0;
    moveListLens[0] = 0;
    moveListAlive[0] = 0;
    bool returnedFromDeeper = false;

    // Make it so that pla is always the defender
    FastBoard::vertex_t pla = state->board.get_state(str_vtx);
    FastBoard::vertex_t opp = static_cast<FastBoard::vertex_t>(FLIP_COLOR(pla));
    const auto turn_color = state->board.get_to_move();

    while (true) {
        // Returned from the root - so that's the answer
        if (stackIdx < 0) {
            if (returnValue == ALIVE) {
                return std::max(max_depth, 1);
            } else {
                if (dead_depth > 0) {
                    // Unable to escape from ladder.
                    return -dead_depth;
                } else {
                    return 0;
                }
            }
        }
        // If we hit the stack limit, just consider it a failed ladder.
        // If we hit a total node count limit, then just assume it doesn't work.
        if (stackIdx >= stackSize - 1
            || searchNodeCount >= MAX_LADDER_SEARCH_NODE_BUDGET) {
            return 0;
        }

        bool isDefender;
        if (turn_color == pla) {
           isDefender = ((stackIdx % 2) == 0);
        } else {
           isDefender = ((stackIdx % 2) == 1);
        }

        // We just entered this level?
        if (moveListCur[stackIdx] == -1) {
            if (state->board.get_state(str_vtx) == FastBoard::EMPTY) {
                returnValue = DEAD;
                if (stackIdx > dead_depth) {
                    // Escape stones are captured.
                    dead_depth = stackIdx;
                }
                returnedFromDeeper = true;
                stackIdx--;
                continue;
            }
            int libs = state->board.get_liberties(str_vtx);
            // Otherwise we need to keep searching.
            // Generate the move list. Attacker and defender generate moves on the group's liberties, but only the defender
            // generates moves on surrounding capturable opposing groups.
            int start = moveListStarts[stackIdx];
            int moveListLen = 0;

            // Defender check
            if (isDefender) {
                // If we are the defender and the group has 2 liberties, we already win.
                if (libs >= 2) {
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
                // Check if can capture the stone of the surrounding opponent.
                int capture_checked[FastBoard::NUM_VERTICES] = {};
                int breath_checked[FastBoard::NUM_VERTICES] = {};
                auto newpos = str_vtx;
                auto n_vtx = 0;
                do {
                    // Check whether can capture the stone at the breathing point of opponent's stones.
                    for (auto d = 0; d < 4; d++) {
                        n_vtx = state->board.get_state_neighbor(newpos, d);
                        if (state->board.get_state(n_vtx) != opp ||
                            state->board.get_liberties(n_vtx) != 1) {
                            continue;
                        }
                        if (capture_checked[state->board.get_parent_stone(n_vtx)]) {
                            continue;
                        }
                        capture_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
                        auto capture_liberty_pos = state->board.get_liberty_pos(1, n_vtx);
                        if (!breath_checked[capture_liberty_pos[0]]) {
                            breath_checked[capture_liberty_pos[0]] = CHECKED;
                            if (state->is_move_legal(pla, capture_liberty_pos[0])) {
                                if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                                    return 0;
                                }
                                buf[start + moveListLen] = capture_liberty_pos[0];
                                moveListLen++;
                            }
                        }
                    }
                    newpos = state->board.get_next_stone(newpos);
                } while (newpos != str_vtx);

                // Check for escape routes.
                auto escape_liberty_pos = state->board.get_liberty_pos(1, str_vtx);
                if (!breath_checked[escape_liberty_pos[0]]
                    && state->is_move_legal(pla, escape_liberty_pos[0])) {
                    breath_checked[escape_liberty_pos[0]] = CHECKED;
                    if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                        return 0;
                    }
                    buf[start + moveListLen] = escape_liberty_pos[0];
                    moveListLen++;
                }

                // Check if the target stones have escaped at this point.
                if (breath_checked[escape_liberty_pos[0]]) {
                    state->play_move(pla, escape_liberty_pos[0]);
                    // Defender immediately wins if there are provably enough libs
                    if (state->board.get_liberties(escape_liberty_pos[0]) >= 3) {
                        state->undo_move();
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                    // Attacker immediately wins if defender has not enough libs and there are no alternatives
                    if (moveListLen == 1 &&
                        state->board.get_liberties(escape_liberty_pos[0]) <= 1) {
                        auto capture_liberty_pos =
                            state->board.get_liberty_pos(1, escape_liberty_pos[0]);
                        if (state->is_move_legal(opp, capture_liberty_pos[0])) {
                            state->undo_move();
                            returnValue = DEAD;
                            if (stackIdx + 1 > dead_depth) {
                                // The only escape route is impossible to move.
                                dead_depth = stackIdx + 1;
                            }
                            returnedFromDeeper = true;
                            stackIdx--;
                            continue;
                        }
                    }
                    state->undo_move();

                    int lowerBoundLibs;
                    int upperBoundLibs;
                    state->board.get_bound_num_liberties_after_play(
                        escape_liberty_pos[0],
                        pla,
                        lowerBoundLibs,
                        upperBoundLibs);
                    // Defender immediately wins if there are provably enough libs
                    if (lowerBoundLibs >= 3) {
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                    // Attacker immediately wins if defender has not enough libs and there are no alternatives
                    if (moveListLen == 1 && upperBoundLibs <= 1) {
                        returnValue = DEAD;
                        if (stackIdx + 1 > dead_depth) {
                            // Upper bound liberties <= 1.
                            dead_depth = stackIdx + 1;
                        }
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                }

                // Is there any way to escape?
                if (moveListLen < 1) {
                    returnValue = DEAD;
                    if (stackIdx + 1 > dead_depth) {
                        // Nothing escape routes.
                        dead_depth = stackIdx + 1;
                    }
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }

            // Attacker check
            } else {
                // If we are the attacker and the group has only 1 liberty, we already win.
                if (libs <= 1) {
                    returnValue = DEAD;
                    if (stackIdx + 1 > dead_depth) {
                        // Escape route is less than one.
                        dead_depth = stackIdx + 1;
                    }
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                } else if (libs >= 3) {
                    returnValue = ALIVE;
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
                auto chase_liberty_pos = state->board.get_liberty_pos(2, str_vtx);
                auto net_pos = 0;
                if (chase_liberty_pos[0] > chase_liberty_pos[1]) {
                    if (chase_liberty_pos[0] - BOARD_SIZE == chase_liberty_pos[1]) {
                        if (state->board.get_state(chase_liberty_pos[0] + 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[0] + 1;
                        } else if (state->board.get_state(chase_liberty_pos[1] - 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[1] - 1;
                        }
                    } else if (chase_liberty_pos[0] - BOARD_SIZE - 2 == chase_liberty_pos[1]) {
                        if (state->board.get_state(chase_liberty_pos[0] - 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[0] - 1;
                        } else if (state->board.get_state(chase_liberty_pos[1] + 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[1] + 1;
                        }
                    }
                } else {
                    if (chase_liberty_pos[1] - BOARD_SIZE == chase_liberty_pos[0]) {
                        if (state->board.get_state(chase_liberty_pos[1] + 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[1] + 1;
                        } else if (state->board.get_state(chase_liberty_pos[0] - 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[0] - 1;
                        }
                    } else if (chase_liberty_pos[1] - BOARD_SIZE - 2 == chase_liberty_pos[0]) {
                        if (state->board.get_state(chase_liberty_pos[1] - 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[1] - 1;
                        } else if (state->board.get_state(chase_liberty_pos[0] + 1)
                            == FastBoard::EMPTY) {
                            net_pos = chase_liberty_pos[0] + 1;
                        }
                    }
                }
                if (net_pos && state->is_move_legal(opp, net_pos)) {
                    state->play_move(opp, net_pos);
                    auto lib0_cnt = 0;
                    auto lib1_cnt = 0;
                    auto atari_cnt = 0;
                    auto n_vtx = 0;
                    for (auto d = 0; d < 4; d++) {
                        n_vtx = state->board.get_state_neighbor(chase_liberty_pos[0], d);
                        if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                        } else if (state->board.get_state(n_vtx) == opp
                            && state->board.get_liberties(n_vtx) == 1) {
                            atari_cnt++;
                        }
                        n_vtx = state->board.get_state_neighbor(chase_liberty_pos[1], d);
                        if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                            lib1_cnt++;
                        } else if (state->board.get_state(n_vtx) == opp
                            && state->board.get_liberties(n_vtx) == 1) {
                            atari_cnt++;
                        }
                    }
                    state->undo_move();
                    if (lib0_cnt == 1 && lib1_cnt == 1 && !atari_cnt) {
                        returnValue = DEAD;
                        if (stackIdx + 5 > dead_depth) {
                            // Both chase routes can be captured.
                            dead_depth = stackIdx + 5;
                        }
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    }
                }
                moveListLen = 2;
                if (static_cast<size_t>(start + moveListLen) >= buf.size()) {
                    return 0;
                }
                buf[start] = chase_liberty_pos[0];
                buf[start + 1] = chase_liberty_pos[1];
                int num_libs0 = 0;
                int num_libs1 = 0;
                auto n_vtx = 0;
                for (auto d = 0; d < 4; d++) {
                    n_vtx = state->board.get_state_neighbor(chase_liberty_pos[0], d);
                    if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                        num_libs0++;
                    }
                    n_vtx = state->board.get_state_neighbor(chase_liberty_pos[1], d);
                    if (state->board.get_state(n_vtx) == FastBoard::EMPTY) {
                        num_libs1++;
                    }
                }
                // If we are the attacker and we're in a double-ko death situation, then assume we win.
                // Both defender liberties must be ko mouths, connecting either ko mouth must not increase the defender's
                // liberties, and none of the attacker's surrounding stones can currently be in atari.
                // This is not complete - there are situations where the defender's connections increase liberties, or where
                // the attacker has stones in atari, but where the defender is still in inescapable atari even if they have
                // a large finite number of ko threats. But it's better than nothing.
                if (num_libs0 == 0 && num_libs1 == 0 &&
                    state->board.would_be_ko_capture(buf[start], opp) &&
                    state->board.would_be_ko_capture(buf[start+1], opp)) {
                    // Both possible moves have no space above, below, left or right
                    // and can only take one stone.
                    auto after_libs = 0;
                    if (state->is_move_legal(opp, chase_liberty_pos[0])) {
                        state->play_move(opp, chase_liberty_pos[0]);
                        after_libs = state->board.get_liberties(chase_liberty_pos[0]);
                        state->undo_move();
                    }
                    if (after_libs <= 2 &&
                        state->is_move_legal(opp, chase_liberty_pos[1])) {
                        state->play_move(opp, chase_liberty_pos[1]);
                        after_libs = state->board.get_liberties(chase_liberty_pos[1]);
                        state->undo_move();
                        if (after_libs <= 2) {
                            // Both moves have 2 or less breathing points after playing.
                            if (!state->board.has_liberty_gaining_captures(str_vtx)) {
                                returnValue = DEAD;
                                if (stackIdx + 1 > dead_depth) {
                                    // Either route will result in a Ko state and
                                    // the escape move will have no way to escape.
                                    dead_depth = stackIdx + 1;
                                }
                                returnedFromDeeper = true;
                                stackIdx--;
                                continue;
                            }
                        }
                    }
                }
                if (buf[start] != buf[start + 1] - (BOARD_SIZE + 1) &&
                    buf[start] != buf[start + 1] - 1 &&
                    buf[start] != buf[start + 1] + 1 &&
                    buf[start] != buf[start + 1] + (BOARD_SIZE + 1)) {
                    // The two breathing points are not adjacent vertically or horizontally.
                    if (num_libs0 >= 3 && num_libs1 >= 3) {
                        // No matter which way move it, it won't be captured.
                        returnValue = ALIVE;
                        returnedFromDeeper = true;
                        stackIdx--;
                        continue;
                    } else if (num_libs0 >= 3) {
                        // If move it into libs0, it may be captured.
                        moveListLen = 1;
                    } else if (num_libs1 >= 3) {
                        // If move it into libs1, it may be captured.
                        buf[start] = buf[start + 1];
                        moveListLen = 1;
                    }
                }
                if (moveListLen > 1) {
                    int num_libsX2 = 0;
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state->board.get_state_neighbor(buf[start], d);
                        if (state->board.get_state(n_vtx) == pla) {
                            int libs = state->board.get_liberties(n_vtx);
                            if (libs > 1) {
                                num_libsX2 += libs * 2 - 3;
                            }
                        }
                    }
                    num_libs0 = num_libs0 * 2 + num_libsX2;
                    num_libsX2 = 0;
                    for (auto d = 0; d < 4; d++) {
                        auto n_vtx = state->board.get_state_neighbor(buf[start + 1], d);
                        if (state->board.get_state(n_vtx) == pla) {
                            int libs = state->board.get_liberties(n_vtx);
                            if (libs > 1) {
                                num_libsX2 += libs * 2 - 3;
                            }
                        }
                    }
                    // Swap the move order
                    num_libs1 = num_libs1 * 2 + num_libsX2;
                    if (num_libs1 > num_libs0) {
                        int tmp = buf[start];
                        buf[start] = buf[start + 1];
                        buf[start + 1] = tmp;
                    }
                }
            }
            moveListLens[stackIdx] = moveListLen;
            moveListAlive[stackIdx] = 0;
            // And indicate to begin search on the first move generated.
            moveListCur[stackIdx] = 0;

        // Else, we returned from a deeper level (or the same level, via illegal move)
        } else {
            // If we returned from deeper we need to undo the move we made
            if (returnedFromDeeper) {
                state->undo_move();
            }
            // Counting the aliving moves
            if (returnValue == ALIVE) {
                moveListAlive[stackIdx]++;
            }
            if (!chase) {
                // Defender has a move that is not ladder captured?
                if (isDefender && returnValue == ALIVE) {
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                // Attacker has a move that does ladder capture?
                } else if (!isDefender && returnValue == DEAD) {
                    returnedFromDeeper = true;
                    stackIdx--;
                    continue;
                }
            }
            // Move on to the next move to search
            moveListCur[stackIdx]++;
        }
        // Alive or dead is determined by the results of trying all moves
        if (moveListCur[stackIdx] >= moveListLens[stackIdx]) {
            if (moveListAlive[stackIdx] < 1) {
                // No move to alive
                if (returnValue == ALIVE) {
                    returnValue = DEAD;
                    if (stackIdx + 1 > dead_depth) {
                        // No move to alive.
                        dead_depth = stackIdx + 1;
                    }
                }
            } else {
                if (isDefender) {
                    // There is more than one move that can alive
                    returnValue = ALIVE;
                } else {
                    if (moveListAlive[stackIdx] == moveListLens[stackIdx]) {
                        // There is no move that can capture an opponent's stone
                        returnValue = ALIVE;
                    } else {
                        // There is more than one move that can capture an opponent's stone
                        if (returnValue == ALIVE) {
                            returnValue = DEAD;
                            if (stackIdx + 1 > dead_depth) {
                                // It can be captured via another route.
                                dead_depth = stackIdx + 1;
                            }
                        }
                    }
                }
            }
            returnedFromDeeper = true;
            stackIdx--;
            continue;
        }
        // Otherwise we do have an next move to search. Grab it.
        int move = buf[moveListStarts[stackIdx] + moveListCur[stackIdx]];
        FastBoard::vertex_t p = (isDefender ? pla : opp);
        // Illegal move - treat it the same as a failed move, but don't return up a level so that we
        // loop again and just try the next move.
        if (!state->is_move_legal(p, move)) {
            if (isDefender) {
                if (returnValue == ALIVE) {
                    returnValue = DEAD;
                    if (stackIdx + 1 > dead_depth) {
                        // Escape move to the place is illegal.
                        dead_depth = stackIdx + 1;
                    }
                }
            } else {
                returnValue = ALIVE;
            }
            returnedFromDeeper = false;
            continue;
        }
        // Play the move!
        state->play_move(p, move);
        searchNodeCount++;
        // And recurse to the next level
        stackIdx++;
        moveListCur[stackIdx] = -1;
        moveListStarts[stackIdx] = moveListStarts[stackIdx-1] + moveListLens[stackIdx-1];
        moveListLens[stackIdx] = 0;
        moveListAlive[stackIdx] = 0;
        if (stackIdx > max_depth) {
            max_depth = stackIdx;
        }
    }
}

int IsLadderChase(
    const GameState* const current_state,
    const int &chase_vtx,
    const GameState* base_state
    )
{
    auto state = std::make_unique<GameState>(current_state);
    std::array<int, 4> str_vtx = {-1, -1, -1, -1};
    std::array<int, 4> liberty_vtx = {-1, -1, -1, -1};
    auto opponent_num = 0;
    const auto opponent_color = state->board.get_to_move();

    // Look for a position where can atari the opponent's stone.
    char ladder_checked[FastBoard::NUM_VERTICES] = {};
    for (auto d = 0; d < 4; d++) {
        auto n_vtx = state->board.get_state_neighbor(chase_vtx, d);
        if (state->board.get_state(n_vtx) == opponent_color
            && !ladder_checked[state->board.get_parent_stone(n_vtx)]
            && state->board.get_liberties(n_vtx) == 1
        ) {
            ladder_checked[state->board.get_parent_stone(n_vtx)] = CHECKED;
            auto liberty_pos = state->board.get_liberty_pos(1, n_vtx);
            if (state->is_move_legal(opponent_color, liberty_pos[0])) {
                str_vtx[opponent_num] = n_vtx;
                liberty_vtx[opponent_num] = liberty_pos[0];
                opponent_num++;
            }
        }
    }
    auto max_depth = 0;
    for (auto opp_i = 0; opp_i < opponent_num; opp_i++) {
        auto ladder_counter = 0;
        auto current_move = str_vtx[opp_i];
        for (int i = base_state->get_movenum(); i >= 0; i -= 2) {
            auto prev_state = base_state->get_game_history()[i];
            auto prev_move = prev_state->get_last_move();
            if (prev_move == str_vtx[opp_i] ||
                state->board.get_parent_stone(prev_move)
                    != state->board.get_parent_stone(current_move)) {
                continue;
            }
            if (prev_state->board.get_liberties(prev_move) == 2) {
                ladder_counter++;
                current_move = prev_move;
            } else {
                break;
            }
        }
        if (ladder_counter >= cfg_offense_stones) {
            state->play_move(opponent_color, liberty_vtx[opp_i]);
            if (state->board.get_liberties(liberty_vtx[opp_i]) == 2) {
                auto depth = IsLadderEscape(state.get(), str_vtx[opp_i], true);
                if (depth > 0) {
                    max_depth = std::max(max_depth, depth + ladder_counter);
                } else {
                    return 0;
                }
            }
            state->undo_move();
        }
    }
    return max_depth;
}

void LadderDetection(
    const GameState* base_state,
    int* const ladder_pos,
    const std::array<float, NUM_INTERSECTIONS> &policy,
    const float &ladder_min_policy,
    const int &check_nodes
    )
{
    auto state = std::make_unique<GameState>(base_state);
    const auto turn_color = state->board.get_to_move();

    if (state->m_komove != FastBoard::NO_VERTEX) {
        return;
    }
    int check_nodes_count = 0;
    for (auto i = 0; i < NUM_INTERSECTIONS; i++) {
        if (check_nodes_count >= check_nodes) {
            break;
        }
        const auto x = i % BOARD_SIZE;
        const auto y = i / BOARD_SIZE;
        const auto vertex = state->board.get_vertex(x, y);
        if (state->board.get_state(vertex) != FastBoard::EMPTY
            || policy[i] < ladder_min_policy
            || !state->is_move_legal(turn_color, vertex)) {
            continue;
        }
        check_nodes_count++;
        auto capture_count = state->board.get_prisoners(turn_color);
        state->play_move(turn_color, vertex);
        capture_count = state->board.get_prisoners(turn_color) - capture_count;
        auto stone_count = state->board.get_string_count(vertex);
        if (cfg_ladder_defense > 0 &&
            state->board.get_liberties(vertex) == 2 &&
            stone_count > capture_count) {

            auto ladder_counter = 0;
            auto current_move = vertex;
            for (int i = base_state->get_movenum() - 1; i >= 0; i -= 2) {
                auto prev_state = base_state->get_game_history()[i];
                auto prev_move = prev_state->get_last_move();
                if (state->board.get_parent_stone(prev_move)
                        != state->board.get_parent_stone(current_move)) {
                    continue;
                }
                if (prev_state->board.get_liberties(prev_move) == 2) {
                    ladder_counter++;
                    current_move = prev_move;
                } else {
                    break;
                }
            }

            if (stone_count >= cfg_defense_stones) {
                auto depth = IsLadderEscape(state.get(), vertex);
                if (depth < 0) {
                    auto move_string = state->move_to_text(vertex);
                    myprintf("can't escape. %s(%s) depth count:%d policy:%f\n",
                        move_string.c_str(),
                        turn_color == FastBoard::WHITE ? "WHITE": "BLACK",
                        depth, policy[i]);
                    ladder_pos[i] = depth - ladder_counter * 2;
                    state->undo_move();
                    continue;
                }
            }
        }
        if (cfg_ladder_offense < 1 || capture_count) {
            state->undo_move();
            continue;
        }
        auto depth = IsLadderChase(state.get(), vertex, base_state);
        if (depth > 0) {
            auto move_string = state->move_to_text(vertex);
            myprintf("shouldn't chase. %s(%s) depth count:%d\n",
                move_string.c_str(),
                turn_color == FastBoard::WHITE ? "WHITE": "BLACK",
                depth);
            ladder_pos[i] = depth;
        }
        state->undo_move();
    }
}
