#ifndef MDPSOLVER_H
#define MDPSOLVER_H

#include <stdexcept>

#include "MDPCore.h"
#include "Policy.h"

template <typename State, typename Action>
class MDPSolver {
protected:

    Policy<State, Action> m_pi; // Policy pi function
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>> m_Q; // Action-value Q function

    MDPCore<State, Action>* m_mdp;

    virtual void initialize_policy() = 0;
    virtual void initialize_value_functions() = 0;
public:
    virtual ~MDPSolver() = default;

    explicit MDPSolver(MDPCore<State, Action>* mdp_core) : m_mdp(mdp_core) {}

    virtual void initialize() {
        initialize_policy();
        initialize_value_functions();
    }

    Action pi(State s)
    {
        return m_pi(s);
    }

    Return v(State s)
    {
        auto it = m_v.find(s);
        if (it == m_v.end()) throw std::runtime_error("Error: Invalid state provided for the v-value function.");

        return it->second;
    }

    Return Q(State s, Action a)
    {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) throw std::runtime_error("Error: Invalid state-action pair provided for the Q-value function.");

        return it->second;
    }

    Policy<State, Action> get_policy() {
        return m_pi;
    }
};

#endif // MDPSOLVER_H
