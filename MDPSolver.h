#pragma once

#include <stdexcept>
#include <tuple>
#include <vector>

#include "MDP.h"

template <typename State, typename Action>
class MDPSolver {
   protected:
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDP<State, Action> &m_mdp;

   public:
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDP<State, Action> &mdp) : m_mdp(mdp) {}

    virtual void initialize() {
        for (const State &s : this->m_mdp.S()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }

    Return v(State s) {
        auto it = m_v.find(s);
        if (it == m_v.end()) {
            throw std::runtime_error("Error: Invalid state provided for the v-value function.");
        }
        return it->second;
    }

    Return Q(State s, Action a) {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end())
            throw std::runtime_error(
                "Error: Invalid state-action pair provided for the Q-value "
                "function.");

        return it->second;
    }
};
