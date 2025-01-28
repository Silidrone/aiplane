#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DeterministicPolicy.h"
#include "ESoftPolicy.h"
#include "TD.h"
#include "taggame/TagGame.h"

static constexpr long double N_OF_STEPS = 1000000000000000000;
static constexpr double POLICY_EPSILON = 0.10f;
static constexpr double TD_ALPHA = 0.1f;

inline int taggame_main() {
    TagGame environment;
    environment.initialize();

    ESoftPolicy<State, Action> policy(POLICY_EPSILON);
    policy.partial_initialize(environment.all_actions());

    TD<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_STEPS, TD_ALPHA);
    mdp_solver.initialize();

    mdp_solver.policy_iteration();

    DeterministicPolicy<State, Action> optimal_policy(policy);

    serialize_to_json(mdp_solver.get_Q(), "taggame-optimal-Q.json");
    serialize_to_json(policy.get_container(), "taggame-stochastic-optimal-policy.json");
    serialize_to_json(optimal_policy.get_container(), "taggame-deterministic-optimal-policy.json");
    return 0;
}
