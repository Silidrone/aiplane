#pragma once

#include <iostream>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "MDP.h"

template <typename State, typename Action>
class MDPSolver {
   protected:
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDP<State, Action> &m_mdp;
    bool m_strict;
    std::mt19937 m_generator;

   public:
    MDPSolver() : m_strict(false), m_generator(std::random_device{}()) {}
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDP<State, Action> &mdp) : m_mdp(mdp) {}

    virtual void initialize() {
        for (const State &s : this->m_mdp.S()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }

        for (const State &s : this->m_mdp.T()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    void set_strict_mode(bool s) { m_strict = s; }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>> get_Q() { return m_Q; }

    Return v(State s) {
        auto it = m_v.find(s);
        if (it == m_v.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error("Error: Invalid state provided for the v-value function.");
        }
        return it->second;
    }

    Return Q(State s, Action a) {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error(
                "Error: Invalid state-action pair provided for the Q-value "
                "function.");
        }

        return it->second;
    }

    std::tuple<Action, Return> Q_best_action(const State &s) {
        try {
            Return max_return = std::numeric_limits<Return>::lowest();
            Action maximizing_action;
            for (Action a : this->m_mdp.A(s)) {
                Return candidate_return = this->Q(s, a);
                if (candidate_return > max_return) {
                    max_return = candidate_return;
                    maximizing_action = a;
                }
            }
            return {maximizing_action, max_return};
        } catch (const std::out_of_range &) {
            return Q_best_action_fallback(s);
        }
    }

    // This is the fallback used when the A(s) state-action space is not pre-initialized.
    std::tuple<Action, Return> Q_best_action_fallback(const State &s) {
        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;

        for (const auto &[state_action, value] : this->m_Q) {
            if (state_action.first == s) {
                if (value > max_return) {
                    max_return = value;
                    maximizing_action = state_action.second;
                }
            }
        }

        return {maximizing_action, max_return};
    }

    Action random_action(const State &s) {
        auto actions = this->m_mdp.A(s);
        if (!actions.empty()) {
            std::uniform_int_distribution<int> dist(0, actions.size() - 1);
            return actions[dist(m_generator)];
        }

        return random_action_fallback(s);  // Use fallback if no actions are available
    }

    Action random_action_fallback(const State &s) {
        std::vector<Action> available_actions;

        for (const auto &[state_action, value] : this->m_Q) {
            if (state_action.first == s) {
                available_actions.push_back(state_action.second);
            }
        }

        if (available_actions.empty()) {
            throw std::runtime_error("random_action_fallback: No actions found in Q-table for the given state.");
        }

        std::uniform_int_distribution<int> dist(0, available_actions.size() - 1);
        return available_actions[dist(m_generator)];
    }

    std::unordered_map<State, Action, StateHash<State>> get_optimal_policy() {
        std::unordered_map<State, Action, StateHash<State>> optimal_policy;

        for (const auto &[state_action, _] : this->m_Q) {
            const State &s = state_action.first;
            if (optimal_policy.find(s) == optimal_policy.end()) {
                auto [best_action, _] = this->Q_best_action(s);
                optimal_policy[s] = best_action;
            }
        }

        return optimal_policy;
    }

    bool load_Q_from_file(std::string file_path) {
        if (!std::filesystem::exists(file_path)) return false;

        std::ifstream q_input(file_path);
        nlohmann::json q_data;
        q_input >> q_data;

        for (const auto &[state_action_str, value] : q_data.items()) {
            State state;
            Action action;

            sscanf(state_action_str.c_str(), "([%d, %d], [%d, %d], %d), (%d, %d)", &std::get<0>(std::get<0>(state)),
                   &std::get<1>(std::get<0>(state)), &std::get<0>(std::get<1>(state)), &std::get<1>(std::get<1>(state)),
                   &std::get<2>(state), &std::get<0>(action), &std::get<1>(action));

            this->set_q(state, action, value);
        }

        return true;
    }
};
