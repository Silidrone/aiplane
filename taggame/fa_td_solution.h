#pragma once

#include <matplot/matplot.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "FA_TD.h"
#include "FunctionApproximator.h"
#include "MDPSolver.h"
#include "Policy.h"
#include "ValueStrategy.h"
#include "m_utils.h"
#include "serialization.h"
#include "taggame/TagGame.h"

constexpr double DISCOUNT_RATE = 1;
static constexpr long double N_OF_EPISODES = 50000000;
static constexpr double POLICY_EPSILON = 0.1;
static constexpr double TD_ALPHA = 0.01;
static const std::string WEIGHTS_FILE = "taggame_fa_weights.json";
static const std::string POLICY_FILE = "fa_td_taggame_optimal_policy.json";

inline int taggame_main() {
    TagGame environment;
    environment.initialize();

    std::function<std::vector<double>(const State&, const Action&)> feature_extractor = [](const State& s,
                                                                                           const Action& a) {
        const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = s;
        const auto& [action_x, action_y] = a;

        std::vector<double> features;

        // Calculate raw direction and distance to predator
        double dx = (my_pos.first - tag_pos.first);
        double dy = (my_pos.second - tag_pos.second);
        double distance = std::sqrt(dx * dx + dy * dy);
        double normalized_distance = distance / MAX_DISTANCE;

        // Normalized direction away from predator (unit vector)
        double dir_magnitude = std::max(0.0001, distance);  // Avoid division by zero
        double dir_x = dx / dir_magnitude;
        double dir_y = dy / dir_magnitude;

        // Normalized action
        double action_magnitude = std::max(0.0001, std::sqrt(action_x * action_x + action_y * action_y));
        double norm_action_x = action_x / MAX_VELOCITY;
        double norm_action_y = action_y / MAX_VELOCITY;
        double action_sign_x = (action_x > 0) ? 1.0 : ((action_x < 0) ? -1.0 : 0.0);
        double action_sign_y = (action_y > 0) ? 1.0 : ((action_y < 0) ? -1.0 : 0.0);

        // Moving away from predator (-1 to 1) - dot product of normalized vectors
        double moving_away = dir_x * norm_action_x + dir_y * norm_action_y;

        // Boundary awareness features
        double dist_left = my_pos.first;
        double dist_right = MAX_X - my_pos.first;
        double dist_top = my_pos.second;
        double dist_bottom = MAX_Y - my_pos.second;

        // Distance to walls × action components
        double left_wall_action = (dist_left / MAX_X) * norm_action_x;
        double right_wall_action = (dist_right / MAX_X) * norm_action_x;
        double top_wall_action = (dist_top / MAX_Y) * norm_action_y;
        double bottom_wall_action = (dist_bottom / MAX_Y) * norm_action_y;

        // Speed difference
        double my_speed = std::sqrt(my_vel.first * my_vel.first + my_vel.second * my_vel.second);
        double tag_speed = std::sqrt(tag_vel.first * tag_vel.first + tag_vel.second * tag_vel.second);
        double speed_diff = (my_speed - tag_speed) / MAX_VELOCITY;
        double speed_diff_action = speed_diff * action_magnitude / MAX_VELOCITY;

        // Direction alignment with predator velocity
        double pred_vel_magnitude = std::max(0.0001, tag_speed);
        double pred_dir_x = tag_vel.first / pred_vel_magnitude;
        double pred_dir_y = tag_vel.second / pred_vel_magnitude;
        double alignment = norm_action_x * pred_dir_x + norm_action_y * pred_dir_y;

        // ALL push_backs in a row at the end
        features.push_back(normalized_distance * action_sign_x);  // Distance to predator × action_x sign
        features.push_back(normalized_distance * action_sign_y);  // Distance to predator × action_y sign
        features.push_back(moving_away);                          // Moving away from predator
        features.push_back(left_wall_action);                     // Distance to left wall × action_x
        features.push_back(right_wall_action);                    // Distance to right wall × action_x
        features.push_back(top_wall_action);                      // Distance to top wall × action_y
        features.push_back(bottom_wall_action);                   // Distance to bottom wall × action_y
        features.push_back(speed_diff_action);                    // Speed difference × action magnitude
        features.push_back(alignment);                            // Direction alignment with predator
        features.push_back(1);                                    // Direction alignment with predator

        return features;
    };

    int feature_dim = 10;  // Total number of features
    auto approximator = new LinearFunctionApproximator<State, Action>(feature_dim, feature_extractor);

    auto value_strategy = new ApproximationValueStrategy<State, Action>();
    value_strategy->initialize(&environment, approximator);

    bool weights_loaded = false;
    try {
        weights_loaded = load_approximator(approximator, output_dir + WEIGHTS_FILE);
        if (weights_loaded) {
            std::cout << "Successfully loaded approximator weights from file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load approximator weights: " << e.what() << std::endl;
    }

    EpsilonGreedyPolicy<State, Action> policy(value_strategy, POLICY_EPSILON);
    FA_TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    try {
        std::cout << "Starting policy iteration..." << std::endl;
        double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });
        std::cout << "Policy iteration completed in " << time_taken << " seconds." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An exception occurred during policy iteration: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "An unknown exception occurred during policy iteration." << std::endl;
    }

    try {
        bool saved = save_approximator(approximator, WEIGHTS_FILE);
        if (saved) {
            std::cout << "Successfully saved approximator weights to file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to save approximator weights: " << e.what() << std::endl;
    }

    // serialize_to_json(policy.optimal(), POLICY_FILE);
    std::cout << "Optimal policy saved to " << POLICY_FILE << std::endl;

    return 0;
}