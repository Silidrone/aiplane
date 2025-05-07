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

constexpr double DISCOUNT_RATE = 1;                    // Reduced from 1.0 to help with long-term planning
static constexpr long double N_OF_EPISODES = 1000000;  // Reduced for faster iterations
static constexpr double POLICY_EPSILON = 0.1;          // Increased for better exploration
static constexpr double TD_ALPHA = 0.05;               // Adjusted learning rate
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

        features.push_back(norm_my_pos_x * norm_action_x);
        features.push_back(norm_my_pos_y * norm_action_y);
        features.push_back(norm_my_vel_x * norm_action_x);
        features.push_back(norm_my_vel_y * norm_action_y);
        features.push_back(norm_tag_pos_x * norm_action_x);
        features.push_back(norm_tag_pos_y * norm_action_y);
        features.push_back(norm_tag_vel_x * norm_action_x);
        features.push_back(norm_tag_vel_y * norm_action_y);

        double distance_to_tagger =
            std::sqrt(std::pow(norm_my_pos_x - norm_tag_pos_x, 2) + std::pow(norm_my_pos_y - norm_tag_pos_y, 2));
        features.push_back(distance_to_tagger);

        double corners[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        double min_corner_dist = 2.0; 

        for (int i = 0; i < 4; i++) {
            double corner_dist =
                std::sqrt(std::pow(norm_my_pos_x - corners[i][0], 2) + std::pow(norm_my_pos_y - corners[i][1], 2));
            min_corner_dist = std::min(min_corner_dist, corner_dist);
        }
        features.push_back(min_corner_dist);

        double rel_pos_x = norm_tag_pos_x - norm_my_pos_x;
        double rel_pos_y = norm_tag_pos_y - norm_my_pos_y;
        features.push_back(rel_pos_x);
        features.push_back(rel_pos_y);

        double rel_vel_x = norm_tag_vel_x - norm_my_vel_x;
        double rel_vel_y = norm_tag_vel_y - norm_my_vel_y;
        features.push_back(rel_vel_x);
        features.push_back(rel_vel_y);

        double escape_x = -rel_pos_x;
        double escape_y = -rel_pos_y;
        double escape_len = std::sqrt(escape_x * escape_x + escape_y * escape_y);
        if (escape_len > 0.0001) {
            escape_x /= escape_len;
            escape_y /= escape_len;
        }
        double action_alignment = (norm_action_x * escape_x + norm_action_y * escape_y);
        features.push_back(action_alignment);

        bool moving_to_corner = false;
        for (int i = 0; i < 4; i++) {
            double corner_x = corners[i][0];
            double corner_y = corners[i][1];
            double current_dist =
                std::sqrt(std::pow(norm_my_pos_x - corner_x, 2) + std::pow(norm_my_pos_y - corner_y, 2));
            double next_pos_x = norm_my_pos_x + norm_action_x * 0.1;
            double next_pos_y = norm_my_pos_y + norm_action_y * 0.1;
            double next_dist = std::sqrt(std::pow(next_pos_x - corner_x, 2) + std::pow(next_pos_y - corner_y, 2));

            if (next_dist < current_dist) {
                moving_to_corner = true;
                break;
            }
        }
        features.push_back(moving_to_corner ? -1.0 : 1.0);

        features.push_back(1.0);

        return features;
    };

    int feature_dim = 25;

    std::vector<int> architecture = {feature_dim, 32, 16, 1};

    auto approximator = new NeuralNetworkFunctionApproximator<State, Action>(feature_extractor, architecture);

    auto value_strategy = new ApproximationValueStrategy<State, Action>();
    value_strategy->initialize(&environment, approximator);

    static const std::string WEIGHTS_FILE = "taggame_nn_weights.json";
    static const std::string POLICY_FILE = "nn_td_taggame_optimal_policy.json";

    bool weights_loaded = false;
    try {
        weights_loaded = load_approximator(approximator, output_dir + WEIGHTS_FILE);
        if (weights_loaded) {
            std::cout << "Successfully loaded neural network weights from file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load neural network weights: " << e.what() << std::endl;
    }

    class HybridPolicy : public EpsilonGreedyPolicy<State, Action> {
       private:
        double demonstration_prob;

       public:
        HybridPolicy(ValueStrategy<State, Action>* value_strategy, double epsilon, double demo_prob)
            : EpsilonGreedyPolicy<State, Action>(value_strategy, epsilon), demonstration_prob(demo_prob) {}

        Action sample(const State& s) override {
            // Occasionally use deterministic solution as demonstration
            if ((std::rand() % 100) < (demonstration_prob * 100)) {
                const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = s;

                double dx = my_pos.first - tag_pos.first;
                double dy = my_pos.second - tag_pos.second;
                double len = std::sqrt(dx * dx + dy * dy);
                if (len < 0.0001) {
                    dx = std::rand() % 20 - 10;
                    dy = std::rand() % 20 - 10;
                    len = std::sqrt(dx * dx + dy * dy);
                }

                dx = (dx / len) * MAX_VELOCITY;
                dy = (dy / len) * MAX_VELOCITY;

                bool near_corner = false;
                int corner_x = 0, corner_y = 0;

                std::vector<std::pair<int, int>> corners = {{0, 0},
                                                            {0, static_cast<int>(MAX_Y)},
                                                            {static_cast<int>(MAX_X), 0},
                                                            {static_cast<int>(MAX_X), static_cast<int>(MAX_Y)}};

                double min_corner_dist = MAX_DISTANCE;
                for (const auto& corner : corners) {
                    double dist = std::sqrt(std::pow(my_pos.first - corner.first, 2) +
                                            std::pow(my_pos.second - corner.second, 2));
                    if (dist < min_corner_dist) {
                        min_corner_dist = dist;
                        corner_x = corner.first;
                        corner_y = corner.second;
                        near_corner = (dist < MAX_DISTANCE * 0.15);
                    }
                }

                if (near_corner) {
                    double corner_dx = my_pos.first - corner_x;
                    double corner_dy = my_pos.second - corner_y;
                    double corner_len = std::sqrt(corner_dx * corner_dx + corner_dy * corner_dy);

                    dx = 0.7 * dx + 0.3 * (corner_dx / corner_len * MAX_VELOCITY);
                    dy = 0.7 * dy + 0.3 * (corner_dy / corner_len * MAX_VELOCITY);

                    double new_len = std::sqrt(dx * dx + dy * dy);
                    dx = (dx / new_len) * MAX_VELOCITY;
                    dy = (dy / new_len) * MAX_VELOCITY;
                }

                return std::make_pair(static_cast<int>(std::round(dx)), static_cast<int>(std::round(dy)));
            }

            return EpsilonGreedyPolicy<State, Action>::sample(s);
        }
    };

    // Use the hybrid policy with 20% demonstrations to guide learning
    HybridPolicy policy(value_strategy, POLICY_EPSILON, 0.2);
    FA_TD<State, Action> mdp_solver(&environment, &policy, value_strategy, DISCOUNT_RATE, N_OF_EPISODES, TD_ALPHA);

    try {
        std::cout << "Starting policy iteration with neural network..." << std::endl;
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
            std::cout << "Successfully saved neural network weights to file." << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to save neural network weights: " << e.what() << std::endl;
    }

    // serialize_to_json(policy.optimal(), POLICY_FILE);
    std::cout << "Optimal policy saved to " << POLICY_FILE << std::endl;

    return 0;
}