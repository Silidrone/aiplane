#pragma once

#include <torch/torch.h>

#include <iostream>
#include <vector>

class SimpleNeuralNetwork : public torch::nn::Module {
   public:
    SimpleNeuralNetwork(int input_size, int hidden_size, int output_size)
        : fc1(register_module("fc1", torch::nn::Linear(input_size, hidden_size))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, hidden_size))),
          fc3(register_module("fc3", torch::nn::Linear(hidden_size, output_size))) {
        torch::nn::init::kaiming_uniform_(fc1->weight);
        torch::nn::init::kaiming_uniform_(fc2->weight);
        torch::nn::init::xavier_uniform_(fc3->weight);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = torch::relu(fc2(x));
        x = fc3(x);
        return x;
    }

    void train_batch(const std::vector<float>& states, const std::vector<float>& targets, float learning_rate = 0.01) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor state_tensor =
            torch::from_blob(const_cast<float*>(states.data()), {1, static_cast<long>(states.size())}, options).clone();

        torch::Tensor target_tensor =
            torch::from_blob(const_cast<float*>(targets.data()), {1, static_cast<long>(targets.size())}, options)
                .clone();

        torch::optim::Adam optimizer(parameters(), learning_rate);

        optimizer.zero_grad();

        torch::Tensor output = forward(state_tensor);

        torch::Tensor loss = torch::mse_loss(output, target_tensor);

        loss.backward();

        optimizer.step();

        std::cout << "Loss: " << loss.item<float>() << std::endl;
    }

    void save_model(const std::string& path) { torch::save(shared_from_this(), path); }

    static std::shared_ptr<SimpleNeuralNetwork> load_model(const std::string& path, int input_size, int hidden_size,
                                                           int output_size) {
        auto model = std::make_shared<SimpleNeuralNetwork>(input_size, hidden_size, output_size);
        torch::load(model, path);
        return model;
    }

   private:
    torch::nn::Linear fc1, fc2, fc3;
};

inline void test_torch_neural_network() {
    try {
        auto model = std::make_shared<SimpleNeuralNetwork>(4, 24, 2);

        std::vector<float> state = {0.5f, -0.2f, 0.1f, 0.8f};
        std::vector<float> target = {1.0f, 0.0f};

        for (int i = 0; i < 10; i++) {
            std::cout << "Training iteration " << i + 1 << std::endl;
            model->train_batch(state, target, 0.01);
        }

        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob(state.data(), {1, static_cast<long>(state.size())}, options).clone();

        torch::Tensor output = model->forward(input);
        std::cout << "Model output: " << output << std::endl;

        model->save_model("./output/test_model.pt");
        std::cout << "Model saved successfully" << std::endl;

        auto loaded_model = SimpleNeuralNetwork::load_model("./output/test_model.pt", 4, 24, 2);
        torch::Tensor loaded_output = loaded_model->forward(input);
        std::cout << "Loaded model output: " << loaded_output << std::endl;

        std::cout << "LibTorch test completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error in LibTorch test: " << e.what() << std::endl;
    }
}