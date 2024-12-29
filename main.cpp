#include <matplot/matplot.h>

#include <chrono>
#include <functional>
#include <iostream>

#include "./barto_sutton_exercises/4_9/GamblersProblemEnvironment.h"
#include "VValuePolicyIteration.h"
#include "ValueIteration.h"
#include "m_utils.h"

void plot_v_f(MDPSolver<State, Action> mdp_solver) {
    matplot::vector_1d x, y;

    for (int i = MIN_STAKE + 1; i < MAX_STAKE; ++i) {
        x.push_back(i);
        y.push_back(static_cast<double>(mdp_solver.v(i)));
    }

    matplot::plot(x, y);

    matplot::xlabel("Amount of money (state)");
    matplot::ylabel("Value function v(s)");
    matplot::title("Value Function for Gambler's Problem");

    matplot::show();
}

int main() {
    GamblersProblemEnvironment environment;
    environment.initialize();

    ValueIteration<State, Action> mdp_solver(environment, DISCOUNT_RATE, POLICY_THRESHOLD_EPSILON);
    mdp_solver.initialize();

    mdp_solver.set_v(MIN_STAKE, 0);
    mdp_solver.set_v(MAX_STAKE, 0);

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    mdp_solver.set_v(MAX_STAKE, 1);

    std::cout << "Time taken: " << time_taken << std::endl;

    auto policy = mdp_solver.get_policy();

    serialize_to_json(policy.map_container(), "policy.json");
    // serialize_to_json(mdp_solver.get_v(), "state-value-function.json");
    environment.plot_policy(policy);
    // plot_v_f(mdp_solver);

    return 0;
}
