#pragma once

#include <stdexcept>
#include <unordered_map>

#include "./m_utils.h"
#include "Policy.h"
#include "StochasticPolicy.h"

template <typename State, typename Action>
class DeterministicPolicy : public Policy<State, Action> {
   protected:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

   public:
    DeterministicPolicy() = default;
    DeterministicPolicy(StochasticPolicy<State, Action>& stochastic_policy) {
        for (const auto& [state, action_probs] : stochastic_policy.get_container()) {
            this->set(state, stochastic_policy.optimal_action(state));
        }
    }

    void set(const State& state, const Action& action) override { m_policy_map[state] = action; }

    Action sample(const State& state) override {
        auto it = m_policy_map.find(state);
        if (it == m_policy_map.end()) {
            Action action = this->fallback_action();
            this->set(state, action);
            return action;
        }
        return it->second;
    }

    virtual Action optimal_action(const State& state) override { return sample(state); }

    void initialize(const std::vector<State>& states,
                    const std::unordered_map<State, std::vector<Action>, StateHash<State>>& actions) override {
        for (const auto& state : states) {
            if (!actions.at(state).empty()) {
                m_policy_map[state] = actions.at(state)[0];
            }
        }
    }

    Action fallback_action() {
        if (this->m_actions.empty()) {
            throw std::runtime_error(
                "Trying to call fallback_action when m_actions is empty! You most likely forgot to call "
                "partial_initialize first.");
        }

        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, this->m_actions.size() - 1);

        return this->m_actions[distribution(generator)];
    }

    const std::unordered_map<State, Action, StateHash<State>>& get_container() const { return m_policy_map; }
};
