#pragma once

#include <unordered_map>
#include <vector>

#include "DeterministicPolicy.h"
#include "MDP.h"
#include "m_types.h"

using State = std::pair<int, int>;
using Action = std::pair<int, int>;

static constexpr Reward CONSTANT_REWARD = -1;
static constexpr int ROW_COUNT = 7;
static constexpr int COL_COUNT = 10;
static const State initial_state = {3, 0};
static const State terminal_state = {3, 7};
static const std::array<int, COL_COUNT> wind = {0, 0, 0, 1, 1, 1, 2, 2, 1, 0};
constexpr std::array<Action, 4> possible_actions = {{
    {-1, 0},  // Up
    {1, 0},   // Down
    {0, -1},  // Left
    {0, 1}    // Right
}};

class WindyGridworld : public MDP<State, Action> {
   protected:
    State walk(const State &s, const Action &a) { return {s.first + a.first, s.second + a.second}; }
    State walk_with_wind(const State &s, const Action &a) {
        State next_state = walk(s, a);
        int new_row = std::max(0, std::min(ROW_COUNT - 1, next_state.first - wind[s.second]));

        return {new_row, next_state.second};
    }
    bool is_valid(const State &s) {
        return s.first >= 0 && s.first < ROW_COUNT && s.second >= 0 && s.second < COL_COUNT;
    };

   public:
    void initialize() override;
    bool is_terminal(const State &s) override;
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;
    void plot_policy(const std::unordered_map<State, Action, StateHash<State>> &);
    void output_trajectory(const std::unordered_map<State, Action, StateHash<State>> &);
};
