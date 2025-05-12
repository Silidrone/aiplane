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

// Define PyTorch neural network model for Q-function approximation
struct QNetwork : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    QNetwork(int input_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, 16))),
          fc2(register_module("fc2", torch::nn::Linear(16, 1))) {
        // Initialize weights
        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::xavier_uniform_(fc2->weight);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }
};

// Memory buffer for experience replay
struct ReplayBuffer {
    std::vector<std::tuple<State, Action, double, State, bool>> buffer;
    size_t capacity;
    std::mt19937 generator;

    ReplayBuffer(size_t size) : capacity(size) {
        buffer.reserve(capacity);
        std::random_device rd;
        generator = std::mt19937(rd());
    }

    void add(const State& state, const Action& action, double reward, const State& next_state, bool done) {
        if (buffer.size() >= capacity) {
            buffer.erase(buffer.begin());
        }
        buffer.emplace_back(state, action, reward, next_state, done);
    }

    std::vector<std::tuple<State, Action, double, State, bool>> sample(size_t batch_size) {
        std::vector<std::tuple<State, Action, double, State, bool>> batch;
        batch_size = std::min(batch_size, buffer.size());

        std::vector<size_t> indices(buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[indices[i]]);
        }

        return batch;
    }

    size_t size() const { return buffer.size(); }
};

// Function to convert state-action pair to tensor
torch::Tensor state_action_to_tensor(const State& state, const Action& action) {
    std::vector<float> features;

    // Normalized state coordinates
    float norm_row = static_cast<float>(state.first) / (ROW_COUNT - 1);
    float norm_col = static_cast<float>(state.second) / (COL_COUNT - 1);

    // Action as one-hot encoding
    std::vector<float> action_encoding(possible_actions.size(), 0.0f);
    for (size_t i = 0; i < possible_actions.size(); i++) {
        if (possible_actions[i] == action) {
            action_encoding[i] = 1.0f;
            break;
        }
    }

    // Fill features vector
    features.push_back(norm_row);
    features.push_back(norm_col);

    // Add action encoding
    for (auto& val : action_encoding) {
        features.push_back(val);
    }

    // Add wind strength
    float wind_strength = static_cast<float>(wind[state.second]) / 2.0f;
    features.push_back(wind_strength);

    // Add distance to goal
    float dist_to_goal = std::abs(state.first - terminal_state.first) + std::abs(state.second - terminal_state.second);
    float norm_dist = dist_to_goal / (ROW_COUNT + COL_COUNT);
    features.push_back(norm_dist);

    // Create tensor from features
    return torch::tensor(features, torch::dtype(torch::kFloat32)).view({1, -1});
}

// PyTorch SARSA implementation
class TorchSARSA {
   private:
    WindyGridworld* env;
    std::shared_ptr<QNetwork> q_network;
    torch::optim::Adam optimizer;
    ReplayBuffer replay_buffer;

    double epsilon;
    double gamma;
    double learning_rate;
    int num_episodes;
    int batch_size;

    // Epsilon-greedy action selection
    Action select_action(const State& state) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(gen) < epsilon) {
            // Random action
            std::vector<Action> valid_actions;
            for (const auto& action : possible_actions) {
                if (env->is_valid(state, action)) {
                    valid_actions.push_back(action);
                }
            }
            if (valid_actions.empty()) {
                return possible_actions[0];  // Fallback
            }
            std::uniform_int_distribution<> action_dis(0, valid_actions.size() - 1);
            return valid_actions[action_dis(gen)];
        } else {
            // Greedy action
            Action best_action = possible_actions[0];
            float best_q_value = std::numeric_limits<float>::lowest();

            for (const auto& action : possible_actions) {
                if (env->is_valid(state, action)) {
                    torch::Tensor state_action = state_action_to_tensor(state, action);
                    float q_value = q_network->forward(state_action).item<float>();

                    if (q_value > best_q_value) {
                        best_q_value = q_value;
                        best_action = action;
                    }
                }
            }

            return best_action;
        }
    }

    // Get best action (greedy policy)
    Action get_best_action(const State& state) {
        Action best_action = possible_actions[0];
        float best_q_value = std::numeric_limits<float>::lowest();

        for (const auto& action : possible_actions) {
            if (env->is_valid(state, action)) {
                torch::Tensor state_action = state_action_to_tensor(state, action);
                float q_value = q_network->forward(state_action).item<float>();

                if (q_value > best_q_value) {
                    best_q_value = q_value;
                    best_action = action;
                }
            }
        }

        return best_action;
    }

    // Update Q-network
    void update_q_network() {
        if (replay_buffer.size() < batch_size) {
            return;
        }

        auto batch = replay_buffer.sample(batch_size);

        for (const auto& experience : batch) {
            const auto& [state, action, reward, next_state, done] = experience;

            torch::Tensor state_action_tensor = state_action_to_tensor(state, action);

            double target_q;
            if (done) {
                target_q = reward;
            } else {
                Action next_action = select_action(next_state);
                torch::Tensor next_state_action_tensor = state_action_to_tensor(next_state, next_action);
                target_q = reward + gamma * q_network->forward(next_state_action_tensor).item<float>();
            }

            torch::Tensor current_q = q_network->forward(state_action_tensor);
            torch::Tensor loss = torch::mse_loss(current_q, torch::tensor(target_q));

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
        }
    }

   public:
    TorchSARSA(WindyGridworld* environment, double eps = 0.1, double discount = 0.99, double lr = 0.01,
               int episodes = 1000, int batch = 16)
        : env(environment),
          q_network(std::make_shared<QNetwork>(8)),  // 2 state dims + 4 action one-hot + wind + dist
          optimizer(q_network->parameters(), torch::optim::AdamOptions(lr)),
          replay_buffer(1000),
          epsilon(eps),
          gamma(discount),
          learning_rate(lr),
          num_episodes(episodes),
          batch_size(batch) {}

    void train() {
        std::cout << "Starting PyTorch SARSA training for WindyGridworld..." << std::endl;

        for (int episode = 0; episode < num_episodes; ++episode) {
            State state = env->reset();
            Action action = select_action(state);
            double episode_reward = 0;

            bool done = false;
            int step = 0;
            while (!done) {
                step++;
                auto [next_state, reward] = env->step(state, action);
                done = env->is_terminal(next_state);

                Action next_action = select_action(next_state);

                // Store experience in replay buffer
                replay_buffer.add(state, action, reward, next_state, done);

                // Update Q-network
                update_q_network();

                state = next_state;
                action = next_action;
                episode_reward += reward;
            }

            // Decay epsilon
            epsilon = std::max(0.01, epsilon * 0.999);

            std::cout << "Episode " << (episode + 1) << " - Reward: " << episode_reward << " - Epsilon: " << epsilon
                      << " step: " << step << std::endl;
        }

        std::cout << "Training completed!" << std::endl;
    }

    // Save the model
    void save_model(const std::string& path) {
        torch::save(q_network, path);
        std::cout << "Model saved to " << path << std::endl;
    }

    // Load the model
    void load_model(const std::string& path) {
        torch::load(q_network, path);
        std::cout << "Model loaded from " << path << std::endl;
    }

    // Extract optimal policy
    std::unordered_map<State, Action, StateHash<State>> extract_policy() {
        std::unordered_map<State, Action, StateHash<State>> policy;

        for (int row = 0; row < ROW_COUNT; ++row) {
            for (int col = 0; col < COL_COUNT; ++col) {
                State state = {row, col};
                if (!env->is_terminal(state)) {
                    policy[state] = get_best_action(state);
                }
            }
        }

        return policy;
    }
};

// Main function for WindyGridworld using PyTorch
inline int windygridworld_main() {
    WindyGridworld environment;
    environment.initialize();

    // Train the agent with smaller, faster parameters
    // Higher learning rate (0.01), fewer episodes (1000), smaller batch size (8)
    TorchSARSA agent(&environment, 0.1, 0.99, 0.01, 1000, 8);

    double time_taken = benchmark([&]() { agent.train(); });
    std::cout << "Time taken: " << time_taken << " seconds" << std::endl << std::endl;

    // Save the model
    agent.save_model("output/windygridworld_torch_model.pt");

    // Extract and visualize the optimal policy
    auto optimal_policy = agent.extract_policy();
    environment.plot_policy(optimal_policy);
    std::cout << std::endl << std::endl;
    environment.output_trajectory(optimal_policy);

    // Save the policy
    serialize_to_json(optimal_policy, "windygridworld-torch-optimal-policy.json");

    return 0;
}