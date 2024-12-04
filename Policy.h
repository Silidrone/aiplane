#ifndef POLICY_H
#define POLICY_H

#include <unordered_map>
#include <stdexcept>

template <typename State, typename Action>
class Policy {
protected:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

public:
    Action operator()(const State& state) const {
        auto it = m_policy_map.find(state);
        if (it == m_policy_map.end()) {
            throw std::runtime_error("Error: Invalid state provided for the pi policy function.");
        }
        return it->second;
    }

    Action& operator[](const State& state) {
        return m_policy_map[state];
    }

    void set(const State& state, const Action& action) {
        m_policy_map[state] = action;
    }

    bool has(const State& state) const {
        return m_policy_map.contains(state);
    }

    const std::unordered_map<State, Action, StateHash<State>>& map_container() const {
        return m_policy_map;
    }
};

#endif // POLICY_H
