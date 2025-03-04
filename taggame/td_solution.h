#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "DerivedStochasticPolicy.h"
#include "DeterministicPolicy.h"
#include "MDPSolver.h"
#include "TD.h"
#include "m_utils.h"
#include "taggame/TagGame.h"

constexpr double DISCOUNT_RATE = 1;
static constexpr long double N_OF_EPISODES = 50000;
static constexpr double POLICY_EPSILON = 0.12;
static constexpr double TD_ALPHA = 0.28;
static const std::string Q_INPUT_FILE = "taggame_q_function.json";
static const std::string POLICY_INPUT_FILE = "taggame_optimal_policy.json";

inline int taggame_main() {
    TagGame environment;
    DerivedStochasticPolicy<State, Action> policy(POLICY_EPSILON);
    TD<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    environment.initialize();
    mdp_solver.initialize();
    mdp_solver.load_Q_from_file(output_dir + Q_INPUT_FILE);

    try {
        mdp_solver.policy_iteration();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during policy iteration. Ignoring and proceeding: " << e.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred during policy iteration. Ignoring and proceeding." << std::endl;
    }

    serialize_to_json(mdp_solver.get_Q(), Q_INPUT_FILE);
    serialize_to_json(mdp_solver.get_optimal_policy(), POLICY_INPUT_FILE);
    return 0;
}
