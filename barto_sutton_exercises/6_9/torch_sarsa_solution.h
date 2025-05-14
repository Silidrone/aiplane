#pragma once

#include <matplot/matplot.h>
#include <torch/torch.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>

#include "Policy.h"
#include "SARSA.h"
#include "TorchModel.h"
#include "ValueStrategy.h"
#include "WindyGridworld.h"
#include "serialization.h"

class WindyGridworldQNet : public TorchModel {
   private:
    torch::nn::Linear fc1{nullptr}, output{nullptr};

   public:
    WindyGridworldQNet(int input_size, int hidden_size = 32) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        output = register_module("output", torch::nn::Linear(hidden_size, 1));
    }

    torch::Tensor forward(torch::Tensor x) override {
        x = torch::relu(fc1->forward(x));
        return output->forward(x);
    }
};

inline torch::Tensor state_action_to_tensor(const State& state, const Action& action) {
    int total_actions = possible_actions.size();
    std::vector<float> features(ROW_COUNT * COL_COUNT * total_actions, 0.0f);

    int action_idx = 0;
    for (size_t i = 0; i < possible_actions.size(); i++) {
        if (possible_actions[i] == action) {
            action_idx = i;
            break;
        }
    }

    int index = (state.first * COL_COUNT + state.second) * total_actions + action_idx;
    features[index] = 1.0f;

    return torch::tensor(features, torch::dtype(torch::kFloat32)).view({1, -1});
}

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    constexpr int N_OF_EPISODES = 1500;
    constexpr double DISCOUNT_RATE = 0.95;
    constexpr double EPSILON = 0.1;
    constexpr double MIN_EPSILON = 0.01;
    constexpr double DECAY_RATE = 0.999;
    constexpr double LEARNING_RATE = 0.001;
    constexpr int HIDDEN_SIZE = 32;
    constexpr int FEATURE_SIZE = ROW_COUNT * COL_COUNT * possible_actions.size();

    auto model = std::make_shared<WindyGridworldQNet>(FEATURE_SIZE, HIDDEN_SIZE);

    TorchValueStrategy<State, Action> value_strategy(model.get(), state_action_to_tensor, LEARNING_RATE);
    value_strategy.initialize(&environment);

    EpsilonGreedyPolicy<State, Action> policy(&value_strategy, EPSILON, MIN_EPSILON, DECAY_RATE);

    SARSA<State, Action> sarsa_agent(&environment, &policy, &value_strategy, DISCOUNT_RATE, N_OF_EPISODES);

    std::cout << "Starting SARSA training with neural network..." << std::endl;

    double time_taken = benchmark([&]() { sarsa_agent.policy_iteration(); });
    std::cout << "Time taken: " << time_taken << " seconds" << std::endl << std::endl;

    auto optimal_policy = policy.optimal();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    serialize_to_json(optimal_policy, "windygridworld-torch-optimal-policy.json");

    return 0;
}