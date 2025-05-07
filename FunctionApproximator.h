#pragma once
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

template <typename State, typename Action>
class FunctionApproximator {
   public:
    virtual double predict(const State& s, const Action& a) const = 0;
    virtual std::vector<double> gradient(const State& s, const Action& a) const = 0;
    virtual void update(const State& s, const Action& a, double error, double step_size) = 0;
    virtual const std::vector<double>& get_weights() const = 0;
    virtual void set_weights(const std::vector<double>& new_weights) = 0;
    virtual ~FunctionApproximator() = default;
};

template <typename State, typename Action>
class LinearFunctionApproximator : public FunctionApproximator<State, Action> {
   private:
    std::vector<double> weights;
    std::function<std::vector<double>(const State&, const Action&)> feature_extractor;

   public:
    LinearFunctionApproximator(int feature_dim, std::function<std::vector<double>(const State&, const Action&)> fe)
        : weights(feature_dim, 0.0), feature_extractor(fe) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dis(gen);
        }
    }

    double predict(const State& s, const Action& a) const override {
        auto x = feature_extractor(s, a);
        double value = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            value += weights[i] * x[i];
        }
        return value;
    }

    std::vector<double> gradient(const State& s, const Action& a) const override {
        return feature_extractor(s, a);  // for linear FA, gradient = features
    }

    void update(const State& s, const Action& a, double error, double step_size) override {
        auto x = feature_extractor(s, a);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] += step_size * error * x[i];
    }

    const std::vector<double>& get_weights() const override { return weights; }

    void set_weights(const std::vector<double>& new_weights) override {
        if (weights.size() != new_weights.size()) {
            throw std::invalid_argument("Weight vector size mismatch");
        }
        weights = new_weights;
    }
};

template <typename State, typename Action>
class NeuralNetworkFunctionApproximator : public FunctionApproximator<State, Action> {
   private:
    std::vector<double> weights;
    std::function<std::vector<double>(const State&, const Action&)> feature_extractor;
    std::vector<int> layer_sizes;

    // Helper function to compute layer outputs and cached activations
    std::vector<std::vector<double>> forward(const std::vector<double>& input) const {
        std::vector<std::vector<double>> activations;
        activations.push_back(input);  // Input layer activations

        size_t weight_idx = 0;

        // Process each hidden layer
        for (size_t l = 0; l < layer_sizes.size() - 1; ++l) {
            const int input_size = layer_sizes[l];
            const int output_size = layer_sizes[l + 1];

            std::vector<double> layer_output(output_size, 0.0);

            // Compute weighted sum for each neuron
            for (int j = 0; j < output_size; ++j) {
                // Add bias term
                layer_output[j] = weights[weight_idx++];

                // Add weighted inputs
                for (int i = 0; i < input_size; ++i) {
                    layer_output[j] += weights[weight_idx++] * activations.back()[i];
                }

                // Apply activation function (ReLU) for all but the last layer
                if (l < layer_sizes.size() - 2) {
                    layer_output[j] = std::max(0.0, layer_output[j]);  // ReLU
                }
            }

            activations.push_back(layer_output);
        }

        return activations;
    }

   public:
    NeuralNetworkFunctionApproximator(std::function<std::vector<double>(const State&, const Action&)> fe,
                                      const std::vector<int>& architecture)
        : feature_extractor(fe), layer_sizes(architecture) {
        if (architecture.size() < 2) {
            throw std::invalid_argument("Network architecture must have at least input and output layers");
        }

        int total_weights = 0;
        for (size_t i = 0; i < architecture.size() - 1; ++i) {
            total_weights += architecture[i + 1] * (architecture[i] + 1);  // weights + biases
        }

        weights.resize(total_weights);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.1, 0.1);

        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dis(gen);
        }
    }

    double predict(const State& s, const Action& a) const override {
        auto features = feature_extractor(s, a);
        auto activations = forward(features);
        return activations.back()[0];  // Output layer has a single neuron
    }

    std::vector<double> gradient(const State& s, const Action& a) const override {
        auto features = feature_extractor(s, a);
        auto activations = forward(features);

        // Backpropagation to compute gradients
        std::vector<double> gradients(weights.size(), 0.0);
        std::vector<std::vector<double>> deltas(layer_sizes.size());

        // Output layer delta: for MSE loss, delta = 1.0 (derivative of identity function)
        // We only compute gradient direction, not the full error gradient
        deltas.back().push_back(1.0);

        // Compute deltas for hidden layers
        for (int l = layer_sizes.size() - 2; l >= 0; --l) {
            int current_size = layer_sizes[l];
            int next_size = layer_sizes[l + 1];
            deltas[l].resize(current_size, 0.0);

            size_t weight_idx = 0;
            for (int i = 0; i < l; ++i) {
                weight_idx += layer_sizes[i + 1] * (layer_sizes[i] + 1);
            }
            weight_idx += next_size;  // Skip biases

            // Compute delta for each neuron in current layer
            for (int i = 0; i < current_size; ++i) {
                for (int j = 0; j < next_size; ++j) {
                    double weight = weights[weight_idx + j * current_size + i];
                    deltas[l][i] += weight * deltas[l + 1][j];
                }

                // Apply derivative of ReLU: 1 if input > 0, else 0
                if (l > 0) {  // Not for input layer
                    if (activations[l][i] <= 0) {
                        deltas[l][i] = 0.0;
                    }
                }
            }
        }

        // Compute weight gradients
        size_t weight_idx = 0;
        for (size_t l = 0; l < layer_sizes.size() - 1; ++l) {
            int input_size = layer_sizes[l];
            int output_size = layer_sizes[l + 1];

            for (int j = 0; j < output_size; ++j) {
                // Bias gradient
                gradients[weight_idx++] = deltas[l + 1][j];

                // Weight gradients
                for (int i = 0; i < input_size; ++i) {
                    gradients[weight_idx++] = deltas[l + 1][j] * activations[l][i];
                }
            }
        }

        return gradients;
    }

    void update(const State& s, const Action& a, double error, double step_size) override {
        auto grads = gradient(s, a);
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] += step_size * error * grads[i];
        }
    }

    const std::vector<double>& get_weights() const override { return weights; }

    void set_weights(const std::vector<double>& new_weights) override {
        if (weights.size() != new_weights.size()) {
            throw std::invalid_argument("Weight vector size mismatch");
        }
        weights = new_weights;
    }
};