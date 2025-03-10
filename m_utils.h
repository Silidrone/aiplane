#pragma once

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "m_types.h"

static const std::string output_dir = "output/";

double pow(double b, int p);
double next_poisson(double lambda);
double poisson_probability(int k, double lambda);
int random_value(int, int);
double random_value(double, double);

// Benchmark function
template <typename Func>
double benchmark(Func&& func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end_time - start_time).count();
}

// ExtractInnerType definition
template <typename T>
struct ExtractInnerType {
    using type = T;
};

template <typename T>
struct ExtractInnerType<std::vector<T>> {
    using type = T;
};

namespace std {
template <>
struct hash<std::pair<int, int>> {
    size_t operator()(const std::pair<int, int>& action) const {
        size_t seed = 0;

        seed ^= std::hash<int>()(action.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(action.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        return seed;
    }
};
}  // namespace std

template <typename State>
struct StateHash {
    size_t operator()(const State& state) const {
        if constexpr (std::is_same_v<State, std::vector<typename ExtractInnerType<State>::type>>) {
            size_t result = 0;
            for (const auto& val : state) {
                result ^= std::hash<typename ExtractInnerType<State>::type>()(val) + 0x9e3779b9 + (result << 6) +
                          (result >> 2);
            }
            return result;
        } else if constexpr (std::is_same_v<State, std::tuple<int, int, bool>>) {
            const auto& [a, b, c] = state;
            return std::hash<int>()(a) ^ (std::hash<int>()(b) << 1) ^ (std::hash<bool>()(c) << 2);
        } else if constexpr (std::is_same_v<State, std::tuple<std::pair<int, int>, std::pair<int, int>, int>>) {
            const auto& [vec1, vec2, d] = state;
            size_t result = 0;

            result ^= std::hash<int>()(vec1.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(vec1.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            result ^= std::hash<int>()(vec2.first) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<int>()(vec2.second) + 0x9e3779b9 + (result << 6) + (result >> 2);

            result ^= std::hash<int>()(d) + 0x9e3779b9 + (result << 6) + (result >> 2);

            return result;
        } else {
            return std::hash<State>()(state);
        }
    }
};

template <typename T>
std::string key_to_string(const T& key);

template <>
inline std::string key_to_string<int>(const int& key) {
    return std::to_string(key);
}

template <>
inline std::string key_to_string<std::vector<int>>(const std::vector<int>& vec) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << vec[i];
    }
    oss << "]";
    return oss.str();
}

template <>
inline std::string key_to_string<std::tuple<int, int, bool>>(const std::tuple<int, int, bool>& key) {
    const auto& [a, b, c] = key;
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " + (c ? "true" : "false") + ")";
}

template <>
inline std::string key_to_string<std::pair<std::tuple<int, int, bool>, bool>>(
    const std::pair<std::tuple<int, int, bool>, bool>& key) {
    const auto& [state, action] = key;
    const auto& [a, b, c] = state;
    return "(" + std::to_string(a) + ", " + std::to_string(b) + ", " + (c ? "true" : "false") + ")," +
           (action ? "hit" : "stick");
}

template <>
inline std::string key_to_string<std::tuple<std::pair<int, int>, std::pair<int, int>, int>>(
    const std::tuple<std::pair<int, int>, std::pair<int, int>, int>& key) {
    const auto& [vec1, vec2, dist] = key;

    auto vec_to_string = [](const std::pair<int, int>& vec) {
        return "(" + std::to_string(static_cast<int>(vec.first)) + ", " + std::to_string(static_cast<int>(vec.second)) +
               ")";
    };

    return "{" + vec_to_string(vec1) + ", " + vec_to_string(vec2) + ", " + std::to_string(dist) + "}";
}

template <>
inline std::string key_to_string<std::pair<int, int>>(const std::pair<int, int>& key) {
    return "(" + std::to_string(key.first) + ", " + std::to_string(key.second) + ")";
}

template <>
inline std::string key_to_string<std::pair<std::pair<int, int>, std::pair<int, int>>>(
    const std::pair<std::pair<int, int>, std::pair<int, int>>& key) {
    return "((" + std::to_string(key.first.first) + ", " + std::to_string(key.first.second) + "), (" +
           std::to_string(key.second.first) + ", " + std::to_string(key.second.second) + "))";
}

template <typename State, typename Action>
struct StateActionPairHash {
    size_t operator()(const std::pair<State, Action>& pair) const {
        return StateHash<State>()(pair.first) ^ (std::hash<Action>()(pair.second) << 1);
    }
};
template <typename Key, typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<Key, Value, Hash>& map, const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        j[key_to_string(key)] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<std::tuple<int, int, bool>, Value, Hash>& map,
                       const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [a, b, c] = key;
        j[key_to_string(a) + "," + key_to_string(b) + "," + (c ? "true" : "false")] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(const std::unordered_map<std::pair<std::tuple<int, int, bool>, bool>, Value, Hash>& map,
                       const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [state, action] = key;
        const auto& [a, b, c] = state;
        j["(" + key_to_string(a) + "," + key_to_string(b) + "," + (c ? "true" : "false") + ")," +
          (action ? "hit" : "stick")] = value;
    }
    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }
    file << j.dump(4);
    file.close();
}

template <typename Value, typename Hash>
void serialize_to_json(
    const std::unordered_map<std::pair<std::tuple<std::pair<int, int>, std::pair<int, int>, int>, std::pair<int, int>>,
                             Value, Hash>& map,
    const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [state, action] = key;
        const auto& [vec1, vec2, integer] = state;

        std::string key_string = "([" + std::to_string(vec1.first) + ", " + std::to_string(vec1.second) + "], " + "[" +
                                 std::to_string(vec2.first) + ", " + std::to_string(vec2.second) + "], " +
                                 std::to_string(integer) + "), " + "(" + std::to_string(action.first) + ", " +
                                 std::to_string(action.second) + ")";

        j[key_string] = value;
    }

    std::ofstream file(output_dir + filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing JSON.");
    }

    file << j.dump(4);
    file.close();
}

namespace std {
template <typename T>
std::string to_string(const std::vector<T>& vec);
}

template <typename State, typename Action>
class MDP;

template <typename State, typename Action>
class Policy;

template <typename State, typename Action>
std::vector<std::tuple<State, Action, Reward>> generate_episode(MDP<State, Action>&, Policy<State, Action>*);
