#include "MC_FV.h"

#include <limits>

#include "m_utils.h"

template <typename State, typename Action>
MC_FV<State, Action>::MC_FV(MDP<State, Action> &mdp_core, const double discount_rate, const double number_of_episodes,
                            const double epsilon)
    : GPI<State, Action>(mdp_core, discount_rate, number_of_episodes), m_epsilon(epsilon) {}

template <typename State, typename Action>
void MC_FV<State, Action>::policy_iteration() {
    Return delta;
    bool policy_stable;
    do {
        auto episode = generate_episode(this->m_mdp, this->m_policy);
        std::unordered_map<std::pair<State, Action>, bool, StateActionPairHash<State, Action>> visited;

        Return G = 0;
        for (auto step : episode) {
            auto [s, a, r, sprime] = step;
            G = this->m_discount_rate * G + r;
            auto first_visit = visited.find({s, a}) == visited.end();
            if (first_visit) {
                N[{s, a}]++;
                auto error = r - this->m_Q[{s, a}];
                this->m_Q[{s, a}] += error / N[{s, a}];
            }
        }
    } while (delta > this->m_policy_threshold);
}

template class MC_FV<std::vector<int>, int>;
template class MC_FV<int, int>;
