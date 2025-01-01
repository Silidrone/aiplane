#pragma once

#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action>
class StochasticPolicy : public Policy<State, Action> {
   protected:
    std::unordered_map<State, std::map<Action, double>, StateHash<State>> m_policy_map;

   public:
    StochasticPolicy() {}
    virtual ~StochasticPolicy() = default;

    virtual void initialize(const std::vector<State>& states,
                            const std::unordered_map<State, std::vector<Action>, StateHash<State>>& actions) override {
        for (const State& s : states) {
            const auto& available_actions = actions.at(s);
            if (!available_actions.empty()) {
                double uniform_probability = 1.0 / available_actions.size();
                for (const Action& a : available_actions) {
                    m_policy_map[s][a] = uniform_probability;
                }
            }
        }
    }

    virtual Action sample(const State& state) const override {
        if (m_policy_map.find(state) == m_policy_map.end()) {
            throw std::runtime_error("Error: State not found in policy");
        }

        const auto& action_probs = m_policy_map.at(state);
        double cumulative_probability = 0.0;
        double random = random_value(0.0, 1.0);  // Generate a random value in [0, 1]

        for (const auto& [action, probability] : action_probs) {
            cumulative_probability += probability;
            if (random <= cumulative_probability) {
                return action;
            }
        }

        throw std::runtime_error("Error: Failed to sample action (probabilities may not sum to 1)");
    }

    virtual void normalize(const State& state) {
        auto& action_probs = m_policy_map[state];
        double total_probability =
            std::accumulate(action_probs.begin(), action_probs.end(), 0.0,
                            [](double sum, const std::pair<Action, double>& ap) { return sum + ap.second; });

        if (total_probability > 0) {
            for (auto& [action, probability] : action_probs) {
                probability /= total_probability;
            }
        }
    }

    const std::map<State, std::map<Action, double>>& get_container_impl() const { return m_policy_map; }
};
