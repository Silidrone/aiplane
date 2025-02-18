#pragma once

#include <stdexcept>
#include <unordered_map>

#include "./m_utils.h"
#include "Policy.h"

template <typename State, typename Action>
class DeterministicPolicy : public Policy<State, Action> {
   protected:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

   public:
    DeterministicPolicy() = default;

    void set(const State& state, const Action& action) { m_policy_map[state] = action; }
    const std::unordered_map<State, Action, StateHash<State>>& get_container() const { return m_policy_map; }

    void initialize(MDPSolver<State, Action>* mdp_solver) override {
        auto& mdp_core = mdp_solver->mdp();
        for (const auto& state : mdp_core.S()) {
            auto actions = mdp_core.A(state);
            if (!actions.empty()) {
                m_policy_map[state] = actions[0];
            }
        }
    }

    Action sample(const State& state) override {
        auto it = m_policy_map.find(state);
        if (it == m_policy_map.end()) {
            throw std::runtime_error("Cannot sample an action for an invalid state!");
        }
        return it->second;
    }
};
