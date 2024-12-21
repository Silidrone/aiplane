#include <matplot/matplot.h>

#include <chrono>
#include <functional>
#include <iostream>

#include "VValuePolicyIteration.h"
#include "ValueIteration.h"
#include "barto_sutton_exercises/4_7/CarRentalEnvironment.h"
// #include "./barto_sutton_exercises/4_9/GamblersProblemEnvironment.h"

int main() {
    // GamblersProblemEnvironment environment;
    CarRentalEnvironment environment;
    ValueIteration<State, Action> mdp_solver(&environment, DISCOUNT_RATE, POLICY_THRESHOLD_EPSILON);
    // mdp_solver.set_v(MIN_STAKE, 0);
    // mdp_solver.set_v(MAX_STAKE, 0);
    environment.initialize();
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.value_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    auto policy = mdp_solver.get_policy();

    policy.serialize_to_json("policy.json");
    environment.plot_policy(policy);

    return 0;
}
