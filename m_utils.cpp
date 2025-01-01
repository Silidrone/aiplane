#include "m_utils.h"

#include <MDP.h>
#include <Policy.h>

#include <algorithm>
#include <random>

int random_value(int a, int b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(a, b);

    return dis(gen);
}

double random_value(double a, double b) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(a, b);
    return dis(gen);
}

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
        const double u = random_value(0.0, 1.0);
        p *= u;
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
std::vector<std::tuple<State, Action, Reward>> generate_episode(MDP<State, Action>& mdp,
                                                                Policy<State, Action>* behavior_policy) {
    std::vector<std::tuple<State, Action, Reward>> episode;
    State state = mdp.reset();
    bool done = false;

    while (!done) {
        Action action = behavior_policy->sample(state);
        auto [next_state, reward] = mdp.step(state, action);
        episode.emplace_back(state, action, reward);
        state = next_state;
        done = mdp.is_terminal(state);
    }

    return episode;
}
template std::vector<std::tuple<std::tuple<int, int, bool>, bool, double>> generate_episode(

    MDP<std::tuple<int, int, bool>, bool>&, Policy<std::tuple<int, int, bool>, bool>*);
