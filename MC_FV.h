#pragma once

#include <limits>

#include "DeterministicPolicy.h"
#include "GPI.h"
#include "m_utils.h"

template <typename State, typename Action>
class MC_FV : public GPI<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>> N;
    DeterministicPolicy<State, Action> m_policy;
    std::unordered_map<State, std::vector<Return>, StateHash<State>> m_returns;
    Return avg_returns(const State& s) {
        if (m_returns.find(s) == m_returns.end()) {
            throw std::runtime_error("State not found in returns.");
        }

        const auto& returns = m_returns[s];

        if (returns.empty()) {
            throw std::runtime_error("No returns available for the given state.");
        }

        double total_sum = 0.0;

        for (const auto& r : returns) {
            total_sum += r;
        }

        return total_sum / static_cast<double>(returns.size());
    }

   public:
    MC_FV(MDP<State, Action>& mdp_core, const double discount_rate, const double number_of_episodes)
        : GPI<State, Action>(mdp_core, discount_rate, number_of_episodes){};

    void policy_iteration() override {
        int episode_n = 0;
        do {
            episode_n++;
            auto episode = generate_episode(this->m_mdp, this->m_policy);
            std::unordered_map<std::pair<State, Action>, bool, StateActionPairHash<State, Action>> visited;

            Return G = 0;
            for (auto step : episode) {
                auto [s, a, r] = step;
                G = this->m_discount_rate * G + r;
                auto first_visit = visited.find({s, a}) == visited.end();
                if (first_visit) {
                    N[{s, a}]++;
                    auto error = r - this->m_Q[{s, a}];
                    this->m_Q[{s, a}] += error / N[{s, a}];

                    // for estimation
                    m_returns[s].push_back(G);
                    this->m_v[s] = avg_returns(s);
                }
            }
        } while (episode_n < this->m_policy_threshold);
    }
    void set_policy(DeterministicPolicy<State, Action>& policy) { m_policy = policy; }
    DeterministicPolicy<State, Action> get_policy() const { return m_policy; }
};