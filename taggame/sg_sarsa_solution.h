#pragma once

#include <matplot/matplot.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>

#include "FunctionApproximator.h"
#include "MDPSolver.h"
#include "Policy.h"
#include "SG_SARSA.h"
#include "ValueStrategy.h"
#include "m_utils.h"
#include "serialization.h"
#include "taggame/TagGame.h"

constexpr double DISCOUNT_RATE = 1;
static constexpr long double N_OF_EPISODES = 50000000;
static constexpr double POLICY_EPSILON = 0.15;
static constexpr double TD_ALPHA = 0.001;
static const std::string WEIGHTS_FILE = "taggame_fa_weights.json";
static const std::string POLICY_FILE = "sg_sarsa_taggame_optimal_policy.json";

inline int taggame_main() {
    TagGame environment;
    environment.initialize();

    std::function<std::vector<double>(const State&, const Action&)> feature_extractor = [](const State& s,
                                                                                           const Action& a) {
        const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = s;
        const auto& [action_x, action_y] = a;

        std::vector<double> features;

        double norm_my_pos_x = my_pos.first / MAX_X;
        double norm_my_pos_y = my_pos.second / MAX_Y;
        double norm_my_vel_x = my_vel.first / MAX_VELOCITY;
        double norm_my_vel_y = my_vel.second / MAX_VELOCITY;
        double norm_tag_pos_x = tag_pos.first / MAX_X;
        double norm_tag_pos_y = tag_pos.second / MAX_Y;
        double norm_tag_vel_x = tag_vel.first / MAX_VELOCITY;
        double norm_tag_vel_y = tag_vel.second / MAX_VELOCITY;

        double norm_action_x = action_x / MAX_VELOCITY;
        double norm_action_y = action_y / MAX_VELOCITY;

        features.push_back(norm_my_pos_x);
        features.push_back(norm_my_pos_y);
        features.push_back(norm_my_vel_x);
        features.push_back(norm_my_vel_y);
        features.push_back(norm_tag_pos_x);
        features.push_back(norm_tag_pos_y);
        features.push_back(norm_tag_vel_x);
        features.push_back(norm_tag_vel_y);
        features.push_back(norm_action_x);
        features.push_back(norm_action_y);
        features.push_back(1.0);

        return features;
    };

    int feature_dim = 11;
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
    SG_SARSA<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

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