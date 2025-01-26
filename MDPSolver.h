#pragma once

#include <iostream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "MDP.h"

template <typename State, typename Action>
class MDPSolver {
   protected:
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    // TODO: Perhaps change this map to have the State as key and the value as action-return pair? Think this through as
    // it has both its pros and cons.
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDP<State, Action> &m_mdp;
    bool m_strict;

   public:
    MDPSolver() : m_strict(false) {}
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDP<State, Action> &mdp) : m_mdp(mdp) {}

    virtual void initialize() {
        for (const State &s : this->m_mdp.S()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }

        for (const State &s : this->m_mdp.T()) {
            m_v[s] = 0;
            for (const Action &a : this->m_mdp.A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    void set_strict_mode(bool s) { m_strict = s; }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>> get_Q() { return m_Q; }

    Return v(State s) {
        auto it = m_v.find(s);
        if (it == m_v.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error("Error: Invalid state provided for the v-value function.");
        }
        return it->second;
    }

    Return Q(State s, Action a) {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) {
            if (!m_strict) return 0;

            throw std::runtime_error(
                "Error: Invalid state-action pair provided for the Q-value "
                "function.");
        }

        return it->second;
    }

    std::tuple<Action, Return> Q_best_action(const State &s) {
        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;
        for (Action a : this->m_mdp.A(s)) {
            Return candidate_return = this->Q(s, a);
            if (candidate_return > max_return) {
                max_return = candidate_return;
                maximizing_action = a;
            }
        }
        return {maximizing_action, max_return};
    }
};
