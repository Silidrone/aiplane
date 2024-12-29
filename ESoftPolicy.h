#pragma once

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./m_utils.h"
#include "StochasticPolicy.h"

template <typename State, typename Action>
class ESoftPolicy : StochasticPolicy<State, Action> {
   protected:
    double m_epsilon;

   public:
    ESoftPolicy(double epsilon) : StochasticPolicy<State, Action>(), m_epsilon(epsilon) {}

    void set(const State& state, const Action& action, double probability) {
        StochasticPolicy<State, Action>::set(state, action, probability);
    }

    void softify(const State& state, const Action& best_action) {
        if (this->m_actions_count == 0) {
            throw std::runtime_error("No actions available to softify for the given state.");
        }

        double uniform_probability = m_epsilon / this->m_actions_count;
        double best_action_probability = (1.0 - m_epsilon) + uniform_probability;

        for (auto& [key, probability] : this->m_policy_map) {
            if (key.first == state) {
                if (key.second == best_action) {
                    probability = best_action_probability;
                } else {
                    probability = uniform_probability;
                }
            }
        }

        this->normalize(state);  // Ensure probabilities sum to 1
    }
};
