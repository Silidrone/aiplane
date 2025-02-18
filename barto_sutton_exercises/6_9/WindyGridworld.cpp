#include "WindyGridworld.h"

#include <matplot/matplot.h>

#include "m_utils.h"

void WindyGridworld::initialize() {
    for (int r = 0; r < ROW_COUNT; r++) {
        for (int c = 0; c < COL_COUNT; c++) {
            State s = {r, c};
            m_S.push_back(s);
            for (auto& a : possible_actions) {
                if (is_valid(walk(s, a))) m_A[s].push_back(a);
            }
        }
    }

    int a = 5;
}

bool WindyGridworld::is_terminal(const State& s) { return s == terminal_state; }

State WindyGridworld::reset() { return initial_state; }

std::pair<State, Reward> WindyGridworld::step(const State& state, const Action& action) {
    State resulting_state = walk_with_wind(state, action);

    if (is_terminal(resulting_state)) {
        return {resulting_state, 0};
    }

    return {resulting_state, CONSTANT_REWARD};
}

void WindyGridworld::plot_policy(const std::unordered_map<State, Action, StateHash<State>>& optimal_policy) {
    std::vector<std::vector<std::string>> policy_grid(ROW_COUNT, std::vector<std::string>(COL_COUNT, " "));

    for (int r = 0; r < ROW_COUNT; r++) {
        for (int c = 0; c < COL_COUNT; c++) {
            State s = {r, c};
            if (is_terminal(s)) {
                policy_grid[r][c] = "G";
                continue;
            }

            auto it = optimal_policy.find(s);
            if (it != optimal_policy.end()) {
                Action action = it->second;
                if (action == possible_actions[0]) policy_grid[r][c] = "↑";  // Up
                if (action == possible_actions[1]) policy_grid[r][c] = "↓";  // Down
                if (action == possible_actions[2]) policy_grid[r][c] = "←";  // Left
                if (action == possible_actions[3]) policy_grid[r][c] = "→";  // Right
            } else {
                policy_grid[r][c] = "?";
            }
        }
    }

    for (const auto& row : policy_grid) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

void WindyGridworld::output_trajectory(const std::unordered_map<State, Action, StateHash<State>>& optimal_policy) {
    State current_state = initial_state;

    std::vector<std::vector<std::string>> grid(ROW_COUNT, std::vector<std::string>(COL_COUNT, " "));

    int move_count = 0;
    while (!is_terminal(current_state)) {
        auto it = optimal_policy.find(current_state);
        if (it == optimal_policy.end()) {
            std::cout << "No optimal action found for state: " << current_state.first << ", " << current_state.second
                      << std::endl;
            break;
        }

        Action action = it->second;

        if (action == possible_actions[0]) grid[current_state.first][current_state.second] = "↑";
        if (action == possible_actions[1]) grid[current_state.first][current_state.second] = "↓";
        if (action == possible_actions[2]) grid[current_state.first][current_state.second] = "←";
        if (action == possible_actions[3]) grid[current_state.first][current_state.second] = "→";

        current_state = step(current_state, action).first;
        move_count++;
    }

    grid[terminal_state.first][terminal_state.second] = "G";

    for (const auto& row : grid) {
        for (const auto& cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Move count: " << move_count << std::endl;
}
