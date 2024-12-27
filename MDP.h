#pragma once

#include <m_utils.h>

#include <iostream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "m_types.h"

template <typename State, typename Action>
class MDP {
   public:
    using Transition = std::tuple<State, Reward, Probability>;
    using Dynamics =
        std::unordered_map<std::pair<State, Action>, std::vector<Transition>, StateActionPairHash<State, Action>>;

   protected:
    std::vector<State> m_S;                                                // State space: S
    std::vector<State> m_ST;                                               // Terminal State space: ST
    std::unordered_map<State, std::vector<Action>, StateHash<State>> m_A;  // Action space: A
    Dynamics m_dynamics;                                                   // Dynamics P function (if known)
    State m_current_state;                                                 // Current state of the environment

   public:
    virtual ~MDP() = default;

    MDP() {}

    virtual void initialize() = 0;

    std::vector<State> S() const { return m_S; }
    std::vector<State> ST() const { return m_ST; }
    std::vector<Action> A(const State& s) const { return m_A.at(s); }

    std::vector<Transition> p(const State& s, const Action& a) const { return m_dynamics.at({s, a}); }
    Dynamics dynamics() const { return m_dynamics; }

    virtual State reset() { throw std::logic_error("reset() must be implemented by the specific environment."); }

    virtual std::pair<State, Reward> step(const Action& action) {
        throw std::logic_error("step() must be implemented by the specific environment.");
    }

    virtual bool is_terminal(const State& state) const {
        return std::find(m_ST.begin(), m_ST.end(), state) != m_ST.end();
    }
};
