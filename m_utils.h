#pragma once

#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
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
struct hash<std::tuple<int, int>> {
    size_t operator()(const std::tuple<int, int>& action) const {
        const auto& [a, b] = action;
        size_t seed = 0;

        seed ^= std::hash<int>()(a) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= std::hash<int>()(b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);

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
        } else if constexpr (std::is_same_v<State, std::tuple<Eigen::Vector2d, Eigen::Vector2d, int, bool>>) {
            const auto& [vec1, vec2, dist, tag_changed] = state;
            size_t result = 0;

            for (int i = 0; i < vec1.size(); ++i) {
                result ^= std::hash<double>()(vec1(i)) + 0x9e3779b9 + (result << 6) + (result >> 2);
            }

            for (int i = 0; i < vec2.size(); ++i) {
                result ^= std::hash<double>()(vec2(i)) + 0x9e3779b9 + (result << 6) + (result >> 2);
            }

            result ^= std::hash<int>()(dist) + 0x9e3779b9 + (result << 6) + (result >> 2);
            result ^= std::hash<bool>()(tag_changed) + 0x9e3779b9 + (result << 6) + (result >> 2);

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
inline std::string key_to_string<std::tuple<Eigen::Vector2d, Eigen::Vector2d, int, bool>>(
    const std::tuple<Eigen::Vector2d, Eigen::Vector2d, int, bool>& key) {
    const auto& [vec1, vec2, dist, flag] = key;

    auto vec_to_string = [](const Eigen::Vector2d& vec) {
        return "(" + std::to_string(static_cast<int>(vec.x())) + ", " + std::to_string(static_cast<int>(vec.y())) + ")";
    };

    return "{" + vec_to_string(vec1) + ", " + vec_to_string(vec2) + ", " + std::to_string(dist) + ", " +
           (flag ? "true" : "false") + "}";
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
    const std::unordered_map<std::pair<std::tuple<Eigen::Vector2d, Eigen::Vector2d, int, bool>, std::tuple<int, int>>,
                             Value, Hash>& map,
    const std::string& filename) {
    nlohmann::json j;
    for (const auto& [key, value] : map) {
        const auto& [state, action] = key;
        const auto& [vec1, vec2, integer, boolean] = state;
        const auto& [action_rotation, action_speed] = action;

        std::string key_string = "([" + std::to_string(vec1.x()) + ", " + std::to_string(vec1.y()) + "], " + "[" +
                                 std::to_string(vec2.x()) + ", " + std::to_string(vec2.y()) + "], " +
                                 std::to_string(integer) + ", " + (boolean ? "true" : "false") + "), " + "Action(" +
                                 std::to_string(action_rotation) + ", " + std::to_string(action_speed) + ")";

        j[key_string] = value;
    }

    std::ofstream file(filename);
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
