#ifndef M_UTILS_H
#define M_UTILS_H

#include <cmath>
#include <vector>

double pow(double b, int p);
double next_poisson(double lambda);
double poisson_probability(int k, double lambda);

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

#endif // M_UTILS_H
