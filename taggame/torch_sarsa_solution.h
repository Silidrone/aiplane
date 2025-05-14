#pragma once

#include <matplot/matplot.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>

#include "MDPSolver.h"
#include "Policy.h"
#include "SARSA.h"
#include "TorchModel.h"
#include "ValueStrategy.h"
#include "m_utils.h"
#include "serialization.h"
#include "taggame/TagGame.h"

// Hyperparameters
constexpr double DISCOUNT_RATE = 1.0;
static constexpr long double N_OF_EPISODES = 50000000;
static constexpr double POLICY_EPSILON = 0.15;
static constexpr double MIN_EPSILON = 0.01;
static constexpr double DECAY_RATE = 0.9999;
static constexpr double LEARNING_RATE = 0.001;
static constexpr int HIDDEN_SIZE = 64;
static const std::string MODEL_FILE = "taggame_torch_model.pt";
static const std::string POLICY_FILE = "taggame_torch_optimal_policy.json";

// Neural network model for Tag Game
class TagGameQNet : public TorchModel {
   private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, output{nullptr};

   public:
    TagGameQNet(int input_size, int hidden_size = 64) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        output = register_module("output", torch::nn::Linear(hidden_size, 1));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return output->forward(x);
    }
};

// Global variable to track device usage
static torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

// Feature extractor that converts state-action pairs to tensor inputs
inline torch::Tensor feature_extractor(const State& s, const Action& a) {
    const auto& [my_pos, my_vel, tag_pos, tag_vel, is_tagged] = s;
    const auto& [action_x, action_y] = a;

    std::vector<double> features;

    // Normalize features to [0,1] range
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

    // Distance between player and tagger (additional feature)
    double dx = my_pos.first - tag_pos.first;
    double dy = my_pos.second - tag_pos.second;
    double distance = std::sqrt(dx * dx + dy * dy) / std::sqrt(MAX_X * MAX_X + MAX_Y * MAX_Y);

    // Feature vector
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
    features.push_back(distance);
    features.push_back(is_tagged ? 1.0 : 0.0);
    features.push_back(1.0);  // Bias term

    // Create tensor on CPU, then move to the appropriate device
    return torch::tensor(features, torch::dtype(torch::kFloat32)).view({1, -1}).to(device);
}

inline int taggame_main() {
    TagGame environment;
    environment.initialize();

    constexpr int FEATURE_SIZE = 13;
    auto model = std::make_shared<TagGameQNet>(FEATURE_SIZE, HIDDEN_SIZE);

    TorchValueStrategy<State, Action> value_strategy(model.get(), feature_extractor, LEARNING_RATE);
    value_strategy.initialize(&environment);

    try {
        torch::load(model, output_dir + MODEL_FILE);
        std::cout << "Successfully loaded model from file." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Could not load model: " << e.what() << std::endl;
        std::cout << "Starting with a new model." << std::endl;
    }

    EpsilonGreedyPolicy<State, Action> policy(&value_strategy, POLICY_EPSILON, MIN_EPSILON, DECAY_RATE);

    SARSA<State, Action> mdp_solver(&environment, &policy, &value_strategy, DISCOUNT_RATE, N_OF_EPISODES);

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
        torch::save(model, output_dir + MODEL_FILE);
        std::cout << "Successfully saved the model." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save the model: " << e.what() << std::endl;
    }

    return 0;
}