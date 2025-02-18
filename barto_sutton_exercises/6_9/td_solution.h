#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DerivedStochasticPolicy.h"
#include "TD.h"
#include "WindyGridworld.h"

static constexpr int N_OF_EPISODES = 6000;
static constexpr double DISCOUNT_RATE = 0.9;  // no discounting
static constexpr double EPSILON = 0.1;
static constexpr double ALPHA = 0.1;  // learning rate

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    DerivedStochasticPolicy<State, Action> policy(EPSILON);

    TD<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES, ALPHA);
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl << std::endl;

    auto optimal_policy = mdp_solver.get_optimal_policy();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    serialize_to_json(mdp_solver.get_Q(), "windygridworld-Q.json");
    serialize_to_json(optimal_policy, "windygridworld-optimal-policy.json");

    return 0;
}
