#pragma once

#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>  // Include JSON library
#include <sstream>
#include <vector>

double pow(double b, int p);
double next_poisson(double lambda);
double poisson_probability(int k, double lambda);

static const std::string output_dir = "output/";

// Definition of the benchmark function
template <typename Func>
double benchmark(Func&& func) {
    auto start_time = std::chrono::high_resolution_clock::now();

    func();

    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end_time - start_time;
    return duration.count();
}

template <typename Key, typename Value, typename Hash>
std::string to_json_string(const std::unordered_map<Key, Value, Hash>& map) {
    nlohmann::ordered_json j;

    std::vector<std::pair<Key, Value>> sorted_vector(map.begin(), map.end());

    std::sort(sorted_vector.begin(), sorted_vector.end(),
              [](const std::pair<Key, Value>& a, const std::pair<Key, Value>& b) { return a.first < b.first; });

    for (const auto& entry : sorted_vector) {
        j[std::to_string(entry.first)] = entry.second;
    }

    return j.dump(2);
}

// Template function to serialize unordered_map to JSON file
template <typename Key, typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<Key, Value, Hash>& map, const std::string& filename) {
    std::ofstream file(output_dir + filename);
    if (file.is_open()) {
        file << to_json_string(map);
        file.close();
    } else {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
}

namespace std {
template <typename T>
string to_string(const std::vector<T>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}
}  // namespace std

template <typename T>
struct ExtractInnerType {
    using type = T;
};

template <typename T>
struct ExtractInnerType<std::vector<T>> {
    using type = T;
};

template <typename State>
struct StateHash {
    size_t operator()(const State& state) const {
        if constexpr (std::is_same_v<State, std::vector<typename ExtractInnerType<State>::type>>) {
            size_t result = 0;
            using InnerType = typename ExtractInnerType<State>::type;
            for (const auto& val : state) {
                result ^= std::hash<InnerType>()(val) + 0x9e3779b9 + (result << 6) + (result >> 2);
            }
            return result;
        } else {
            return std::hash<State>()(state);
        }
    }
};

template <typename State, typename Action>
struct StateActionPairHash {
    std::size_t operator()(const std::pair<State, Action>& pair) const {
        const std::size_t hash1 = StateHash<State>()(pair.first);
        const std::size_t hash2 = std::hash<Action>()(pair.second);
        return hash1 ^ (hash2 << 1);
    }
};

#include "m_types.h"

template <typename State, typename Action>
class MDP;

template <typename State, typename Action>
class StochasticPolicy;

template <typename State, typename Action>
std::vector<std::tuple<State, Action, Reward, State>> generate_episode(MDP<State, Action>&,
                                                                       StochasticPolicy<State, Action>);