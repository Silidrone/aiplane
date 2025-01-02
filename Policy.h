#pragma once

#include <unordered_map>
#include <vector>

#include "m_utils.h"

template <typename State, typename Action>
class Policy {
   public:
    virtual ~Policy() = default;

    virtual void set(const State& state, const Action& action) = 0;
    virtual void initialize(const std::vector<State>& states,
                            const std::unordered_map<State, std::vector<Action>, StateHash<State>>& actions) = 0;
    virtual Action sample(const State& state) const = 0;
    virtual Action optimal_action(const State& state) const = 0;
};