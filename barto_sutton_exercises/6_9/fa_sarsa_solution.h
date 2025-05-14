#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "FunctionApproximator.h"
#include "Policy.h"
#include "SARSA.h"
#include "ValueStrategy.h"
#include "WindyGridworld.h"
#include "serialization.h"

static constexpr int N_OF_EPISODES = 6000;
static constexpr double DISCOUNT_RATE = 0.9;
static constexpr double EPSILON = 0.1;
static constexpr double ALPHA = 0.1;

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    std::function<std::vector<double>(const State&, const Action&)> feature_extractor = [](const State& s,
                                                                                           const Action& a) {
        int total_actions = possible_actions.size();
        std::vector<double> features(ROW_COUNT * COL_COUNT * total_actions, 0.0);

        int action_idx = 0;
        for (size_t i = 0; i < possible_actions.size(); i++) {
            if (possible_actions[i] == a) {
                action_idx = i;
                break;
            }
        }

        int index = (s.first * COL_COUNT + s.second) * total_actions + action_idx;
        features[index] = 1.0;
        return features;
    };

    int total_actions = possible_actions.size();
    int feature_dim = ROW_COUNT * COL_COUNT * total_actions;

    LinearFunctionApproximator<State, Action> approximator(feature_dim, feature_extractor);
    ApproximationValueStrategy<State, Action> value_strategy;

    value_strategy.initialize(&environment, &approximator);

    EpsilonGreedyPolicy<State, Action> policy(&value_strategy, EPSILON);

    SARSA<State, Action> mdp_solver(&environment, &policy, &value_strategy, DISCOUNT_RATE, N_OF_EPISODES, ALPHA);

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl << std::endl;

    auto optimal_policy = policy.optimal();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    serialize_to_json(optimal_policy, "windygridworld-optimal-policy.json");

    return 0;
}