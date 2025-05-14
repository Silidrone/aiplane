#pragma once

#include <algorithm>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

/**
 * Experience replay buffer for reinforcement learning.
 * Stores (state, action, reward, next_state, done) tuples and provides
 * sampling functionality for training.
 */
template <typename State, typename Action>
class ReplayBuffer {
   private:
    std::vector<std::tuple<State, Action, double, State, bool>> buffer;
    size_t capacity;  // Maximum number of experiences to store in the buffer
    std::mt19937 generator;

   public:
    ReplayBuffer(size_t size) : capacity(size) {
        buffer.reserve(capacity);
        std::random_device rd;
        generator = std::mt19937(rd());
    }

    /**
     * Adds an experience to the buffer. If the buffer is at capacity,
     * the oldest experience will be removed.
     */
    void add(const State& state, const Action& action, double reward, const State& next_state, bool done) {
        if (buffer.size() >= capacity) {
            buffer.erase(buffer.begin());
        }
        buffer.emplace_back(state, action, reward, next_state, done);
    }

    // Samples a batch of experiences randomly from the buffer.
    std::vector<std::tuple<State, Action, double, State, bool>> sample(size_t batch_size) {
        std::vector<std::tuple<State, Action, double, State, bool>> batch;
        batch_size = std::min(batch_size, buffer.size());

        std::vector<size_t> indices(buffer.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        batch.reserve(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[indices[i]]);
        }

        return batch;
    }

    size_t size() const { return buffer.size(); }
    size_t get_capacity() const { return capacity; }
    void clear() { buffer.clear(); }
};