#include <matplot/matplot.h>

#include <chrono>
#include <functional>
#include <iostream>

#include "ValueIteration.h"
#include "barto_sutton_exercises/4_7/CarRentalEnvironment.h"

static constexpr double DISCOUNT_RATE = 0.9f;
static constexpr long double POLICY_THRESHOLD_EPSILON = 0.1;

inline int car_rental_main() {
    CarRentalEnvironment environment;
    ValueIteration<State, Action> mdp_solver(environment, DISCOUNT_RATE, POLICY_THRESHOLD_EPSILON);
    environment.initialize();
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    auto policy = mdp_solver.get_policy();

    serialize_to_json(policy.get_container(), "car-rental-policy.json");
    serialize_to_json(mdp_solver.get_v(), "car-rental-value-function.json");
    environment.plot_policy(policy);

    return 0;
}