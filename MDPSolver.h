#pragma once

#include <stdexcept>

#include "MDPCore.h"
#include "Policy.h"

template <typename State, typename Action>
class MDPSolver {
   protected:
    Policy<State, Action> m_pi;                                 // Policy pi function
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDPCore<State, Action> *m_mdp;

    virtual void initialize_policy() {
        for (const State &s : this->m_mdp->S()) {
            m_pi.set(s, 0);
        }
    }

    virtual void initialize_value_functions() {
        for (const State &s : this->m_mdp->S()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp->A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

   public:
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDPCore<State, Action> *mdp_core) : m_mdp(mdp_core) {}

    virtual void initialize() {
        initialize_policy();
        initialize_value_functions();
    }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }
    void set_pi(State s, Action a) { m_pi[s] = a; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }

    Action pi(State s) { return m_pi(s); }

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

    Policy<State, Action> get_policy() { return m_pi; }
};
