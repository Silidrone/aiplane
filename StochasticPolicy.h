#pragma once

#include <map>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "./m_utils.h"
#include "MDP.h"

template <typename State, typename Action>
class StochasticPolicy {
   protected:
    std::map<std::pair<State, Action>, double, StateActionPairHash<State, Action>> m_policy_map;
    int m_actions_count;

   public:
    double operator()(const State& state, const Action& action) const {
        auto it = m_policy_map.find({state, action});
        if (it == m_policy_map.end()) {
            throw std::runtime_error("Error: Invalid state-action pair provided for the policy.");
        }
        return it->second;
    }

    std::vector<std::pair<Action, double>> operator()(const State& state) const {
        std::vector<std::pair<Action, double>> action_probabilities;

        for (const auto& [key, probability] : m_policy_map) {
            const auto& [s, action] = key;
            if (s == state) {
                action_probabilities.emplace_back(action, probability);
            }
        }

        return action_probabilities;
    }

    void set(const State& state, const Action& action, double probability) {
        m_policy_map[{state, action}] = probability;
    }

    void normalize(const State& state) {
        double total_probability = 0.0;

        for (const auto& [key, probability] : m_policy_map) {
            const auto& [s, action] = key;
            if (s == state) {
                total_probability += probability;
            }
        }

        if (total_probability > 0) {
            for (auto& [key, probability] : m_policy_map) {
                const auto& [s, action] = key;
                if (s == state) {
                    probability /= total_probability;
                }
            }
        }
    }

    Action sample(const State& state) {
        double random_value = []() {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            return dis(gen);
        }();

        double cumulative_probability = 0.0;
        for (const auto& [key, probability] : m_policy_map) {
            const auto& [s, action] = key;
            if (s == state) {
                cumulative_probability += probability;
                if (random_value <= cumulative_probability) {
                    return action;
                }
            }
        }

        throw std::runtime_error("Error: Failed to sample action");
    }

    void initialize_zero(const std::vector<State>& states,
                         const std::unordered_map<State, std::vector<Action>>& actions) {
        for (const auto& state : states) {
            for (const auto& action : actions.at(state)) {
                m_policy_map[{state, action}] = 0.0;
            }
        }
        m_actions_count = actions.size();
    }

    void initialize_uniform(const std::vector<State>& states,
                            const std::unordered_map<State, std::vector<Action>>& actions) {
        for (const State& s : states) {
            if (!actions.empty()) {
                double uniform_probability = 1.0 / actions.size();
                for (const Action& a : actions) {
                    m_policy_map.set(s, a, uniform_probability);
                }
            }
        }
    }

    const std::unordered_map<std::pair<State, Action>, double, StateActionPairHash<State, Action>>& map_container()
        const {
        return m_policy_map;
    }
};
