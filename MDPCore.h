#ifndef MDPCORE_H
#define MDPCORE_H

#include <vector>
#include <unordered_map>
#include <tuple>
#include <functional>

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

using Reward = double;
using Return = double;
using TimeStep = int;
using Probability = double;

template <typename State, typename Action>
class MDPCore {
public:
    using Transition = std::tuple<State, Reward, Probability>;
    using Dynamics = std::unordered_map<std::pair<State, Action>, std::vector<Transition>, StateActionPairHash<State, Action>>;

protected:
    std::vector<State> m_S;   // State space: S
    std::vector<Action> m_A;  // Action space: A
    Dynamics m_dynamics;      // Dynamics p function
    TimeStep m_T;

    virtual void initialize_state_space() {};
    virtual void initialize_action_space() {}
    virtual void initialize_dynamics() {};

public:
    virtual ~MDPCore() = default;

    MDPCore() : m_T(0) {}

    virtual void initialize() {
        initialize_state_space();
        initialize_action_space();
        initialize_dynamics();
    }

    void increment_time_step() {
        ++m_T;
    }

    [[nodiscard]] TimeStep get_time_step() const {
        return m_T;
    }

    std::vector<Transition> p(State s, Action a) {
        return m_dynamics[{s, a}];
    }

    Dynamics dynamics() {
        return m_dynamics;
    }

    std::vector<State> S() {
        return m_S;
    }

    std::vector<Action> A() {
        return m_A;
    }
};

#endif // MDPCORE_H