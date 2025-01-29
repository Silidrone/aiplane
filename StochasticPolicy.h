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

    std::map<Action, double>& get_safe_action_probs(const State& s) {
        std::map<Action, double>& action_probs = m_policy_map[s];

        if (action_probs.empty()) {
            if (this->m_actions.empty()) {
                throw std::runtime_error(
                    "Trying to call get_safe_action_probs (which means you are dealing with uninitialized states) when "
                    "m_actions is empty! You most likely forgot to call partial_initialize first.");
            }

            for (const auto& action : this->m_actions) {  // equi-probable assignment of actions on new state
                action_probs[action] = 1.0 / this->m_actions.size();
            }
        }

        return action_probs;
    }

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

    virtual Action sample(const State& state) override {
        auto& action_probs = this->get_safe_action_probs(state);

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

    virtual Action optimal_action(const State& state) override {
        auto& action_probs = m_policy_map.at(state);

        if (action_probs.empty()) {
            throw std::runtime_error("optimal_action: No actions available for the given state in the policy map.");
        }

        double highest_probability = 0;
        Action optimal_action;

        for (const auto& [action, probability] : action_probs) {
            if (probability > highest_probability) {
                highest_probability = probability;
                optimal_action = action;
            }
        }

        return optimal_action;
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

    void raw_assign(const State& s, std::map<Action, double> action_probs) {
        m_policy_map[s] = std::move(action_probs);
        normalize(s);
    }

    const std::unordered_map<State, std::map<Action, double>, StateHash<State>>& get_container() const {
        return m_policy_map;
    }
};
