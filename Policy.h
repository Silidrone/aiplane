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
};
