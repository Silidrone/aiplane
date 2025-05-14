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
#include "ValueStrategy.h"
#include "WindyGridworld.h"
#include "serialization.h"

torch::Tensor state_action_to_tensor(const State& state, const Action& action) {
    std::vector<float> features;

    float norm_row = static_cast<float>(state.first) / (ROW_COUNT - 1);
    float norm_col = static_cast<float>(state.second) / (COL_COUNT - 1);

    std::vector<float> action_encoding(possible_actions.size(), 0.0f);
    for (size_t i = 0; i < possible_actions.size(); i++) {
        if (possible_actions[i] == action) {
            action_encoding[i] = 1.0f;
            break;
        }
    }

    features.push_back(norm_row);
    features.push_back(norm_col);

    for (auto& val : action_encoding) {
        features.push_back(val);
    }

    float wind_strength = static_cast<float>(wind[state.second]) / 2.0f;
    features.push_back(wind_strength);

    float dist_to_goal = std::abs(state.first - terminal_state.first) + std::abs(state.second - terminal_state.second);
    float norm_dist = dist_to_goal / (ROW_COUNT + COL_COUNT);
    features.push_back(norm_dist);

    return torch::tensor(features, torch::dtype(torch::kFloat32)).view({1, -1});
}

inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    TorchSARSA agent(&environment, 0.1, 0.99, 0.01, 1000, 8);

    double time_taken = benchmark([&]() { agent.train(); });
    std::cout << "Time taken: " << time_taken << " seconds" << std::endl << std::endl;

    agent.save_model("output/windygridworld_torch_model.pt");

    auto optimal_policy = agent.extract_policy();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    serialize_to_json(optimal_policy, "windygridworld-torch-optimal-policy.json");

    return 0;
}