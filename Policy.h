#pragma once

#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "m_utils.h"

template <typename State, typename Action>
class Policy {
   protected:
    std::vector<Action> m_actions;

   public:
    Policy() = default;
    Policy(const Policy &other) : m_actions(other.m_actions) {}
    virtual ~Policy() = default;

    virtual void set(const State &, const Action &) = 0;
    virtual Action sample(const State &) = 0;
    virtual Action optimal_action(const State &) = 0;

    virtual void initialize(const std::vector<State> &,
                            const std::unordered_map<State, std::vector<Action>, StateHash<State>> &) {
        throw std::runtime_error("initialize() not implemented by this policy.");
    }

    virtual void partial_initialize(const std::vector<Action> &actions) { m_actions = actions; }

    virtual Action fallback_action() {
        if (m_actions.empty()) {
            throw std::runtime_error(
                "Trying to call fallback_action when m_actions is empty! You most likely forgot to call "
                "partial_initialize first.");
        }

        static thread_local std::mt19937 generator(std::random_device{}());
        std::uniform_int_distribution<int> distribution(0, m_actions.size() - 1);

        return m_actions[distribution(generator)];
    }
};
