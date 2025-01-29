#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "DeterministicPolicy.h"
#include "ESoftPolicy.h"
#include "TD.h"
#include "m_utils.h"
#include "taggame/TagGame.h"

static constexpr long double N_OF_EPISODES = 10000;
static constexpr double POLICY_EPSILON = 0.05;
static constexpr double TD_ALPHA = 0.1;
static const std::string Q_INPUT_FILE = "uprightq.json";
static const std::string POLICY_INPUT_FILE = "uprightsp.json";

inline int taggame_main() {
    TagGame environment;
    ESoftPolicy<State, Action> policy(POLICY_EPSILON);
    TD<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    environment.initialize();
    policy.partial_initialize(environment.all_actions());
    // TODO: Refactor later so that it can be accessible in a more generalized context, not just here in taggame_main.
    if (std::filesystem::exists(output_dir + Q_INPUT_FILE)) {
        std::ifstream q_input(output_dir + Q_INPUT_FILE);
        nlohmann::json q_data;
        q_input >> q_data;

        for (const auto& [state_action_str, value] : q_data.items()) {
            State state;
            Action action;

            // Parse state and action from string
            sscanf(state_action_str.c_str(), "([%d, %d], [%d, %d], %d), Action(%d, %d)",
                   &std::get<0>(std::get<0>(state)), &std::get<1>(std::get<0>(state)), &std::get<0>(std::get<1>(state)),
                   &std::get<1>(std::get<1>(state)), &std::get<2>(state), &std::get<0>(action), &std::get<1>(action));

            // Assign Q-value
            mdp_solver.set_q(state, action, value);
        }
    } else {
        mdp_solver.initialize();
    }

    if (std::filesystem::exists(POLICY_INPUT_FILE)) {
        std::ifstream policy_input(POLICY_INPUT_FILE);
        nlohmann::json policy_data;
        policy_input >> policy_data;

        for (const auto& [state_str, actions_probs] : policy_data.items()) {
            State state;
            std::map<Action, double> action_probs;

            // Parse state from string
            sscanf(state_str.c_str(), "{(%d, %d), (%d, %d), %d}", &std::get<0>(std::get<0>(state)),
                   &std::get<1>(std::get<0>(state)), &std::get<0>(std::get<1>(state)), &std::get<1>(std::get<1>(state)),
                   &std::get<2>(state));

            // Parse action-probability pairs
            for (const auto& action_prob : actions_probs) {
                const auto& action_vec = action_prob[0];  // Action vector
                double prob = action_prob[1];             // Corresponding probability

                Action action = {action_vec[0], action_vec[1]};
                action_probs[action] = prob;
            }

            // Assign raw probabilities to the policy
            policy.raw_assign(state, action_probs);
        }
    }

    try {
        mdp_solver.policy_iteration();
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during policy iteration. Ignoring and proceeding: " << e.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred during policy iteration. Ignoring and proceeding." << std::endl;
    }

    serialize_to_json(mdp_solver.get_Q(), Q_INPUT_FILE);
    serialize_to_json(policy.get_container(), POLICY_INPUT_FILE);
    return 0;
}
