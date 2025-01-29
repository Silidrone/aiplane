#pragma once

#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./m_utils.h"
#include "StochasticPolicy.h"

template <typename State, typename Action>
class ESoftPolicy : public StochasticPolicy<State, Action> {
   protected:
    double m_epsilon;

   public:
    ESoftPolicy(double epsilon) : StochasticPolicy<State, Action>(), m_epsilon(epsilon) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
    }

    void set(const State& state, const Action& best_action) override {
        auto& action_probs = this->get_safe_action_probs(state);

        size_t actions_count = action_probs.size();
        double uniform_probability = m_epsilon / actions_count;
        double best_action_probability = (1.0 - m_epsilon) + uniform_probability;

        for (auto& [action, probability] : action_probs) {
            if (action == best_action) {
                probability = best_action_probability;
            } else {
                probability = uniform_probability;
            }
        }

        this->normalize(state);
    }
};
