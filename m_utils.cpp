#include "m_utils.h"

#include <cstdlib>
#include <stdexcept>

#include "MDP.h"
#include "StochasticPolicy.h"

double pow(const double b, const int p) {
    if (p == 0) return 1;
    if (p == 1) return b;

    return b * pow(b, p - 1);
}

double next_poisson(const double lambda) {
    const double L = exp(-lambda);  // e^(-λ)
    double p = 1.0;
    int k = 0;

    do {
        k++;
        const double u = rand() / (RAND_MAX + 1.0);  // Generate uniform random number u between 0 and 1
        p *= u;                                      // Update probability
    } while (p > L);

    return k - 1;  // Subtract 1 because we start from 0 events
}

double poisson_probability(const int k, const double lambda) {
    if (k < 0) {
        throw std::invalid_argument("k must be non-negative.");
    }

    return (pow(lambda, k) * exp(-lambda)) / tgamma(k + 1);
}

template <typename State, typename Action>
std::vector<std::tuple<State, Action, Reward, State>> generate_episode(
    MDP<State, Action>& mdp, StochasticPolicy<State, Action> behavior_policy) {
    std::vector<std::tuple<State, Action, Reward, State>> episode;
    State state = mdp.reset();
    bool done = false;

    while (!done) {
        Action action = behavior_policy.sample(state);
        auto [next_state, reward] = mdp.step(action);
        episode.emplace_back(state, action, reward, next_state);
        state = next_state;
        done = mdp.is_terminal(state);
    }

    return episode;
}

template std::vector<std::tuple<std::vector<int>, int, double, std::vector<int>>>
generate_episode<std::vector<int>, int>(MDP<std::vector<int>, int>&, StochasticPolicy<std::vector<int>, int>);

template std::vector<std::tuple<int, int, double, int>> generate_episode<int, int>(MDP<int, int>&,
                                                                                   StochasticPolicy<int, int>);