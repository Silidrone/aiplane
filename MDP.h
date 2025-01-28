#pragma once

#include <m_utils.h>

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
    std::vector<State> m_T;                                                // Terminal State space: T
    std::unordered_map<State, std::vector<Action>, StateHash<State>> m_A;  // Action space: A
    Dynamics m_dynamics;                                                   // Dynamics P function (if known)
    bool m_is_continuous;

   public:
    virtual ~MDP() = default;

    MDP(bool is_continuous = false) : m_is_continuous(is_continuous) {}

    bool is_continuous() { return m_is_continuous; }

    virtual void initialize() = 0;

    std::vector<State> S() const { return m_S; }
    std::vector<State> T() const { return m_T; }
    std::vector<Action> A(const State& s) const { return m_A.at(s); }
    std::unordered_map<State, std::vector<Action>, StateHash<State>> A() const { return m_A; }

    std::vector<Transition> p(const State& s, const Action& a) const { return m_dynamics.at({s, a}); }
    Dynamics dynamics() const { return m_dynamics; }

    virtual State reset() { throw std::logic_error("The reset function is not available in this environment."); }

    virtual std::pair<State, Reward> step(const State& state, const Action& action) {
        throw std::logic_error("The step function is not available in this environment.");
    }

    virtual bool is_terminal(const State& state) {
        if (m_is_continuous) return false;

        return std::find(m_T.begin(), m_T.end(), state) != m_T.end();
    }
};
