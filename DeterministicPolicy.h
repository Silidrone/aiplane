#pragma once

#include <stdexcept>
#include <unordered_map>

#include "./m_utils.h"

template <typename State, typename Action>
class DeterministicPolicy {
   protected:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

   public:
    Action operator()(const State& state) const {
        auto it = m_policy_map.find(state);
        if (it == m_policy_map.end()) {
            throw std::runtime_error("Error: Invalid state provided for the pi policy function.");
        }
        return it->second;
    }

    Action& operator[](const State& state) { return m_policy_map[state]; }

    void set(const State& state, const Action& action) { m_policy_map[state] = action; }

    const std::unordered_map<State, Action, StateHash<State>>& map_container() const { return m_policy_map; }

    void initialize_with_first_action(const MDP<State, Action>& mdp) {
        for (const auto& state : mdp.S()) {
            for (const auto& act : mdp.A(state)) {
                m_policy_map[state] = act;
                break;
            }
        }
    }
};
