#pragma once

#include <stdexcept>
#include <tuple>
#include <vector>

#include "MDP.h"
#include "Policy.h"

// Note: on-policy algorithms just use the target policy leaving behavior policy
// unchanged. This could have been enforced but I am sacrificing elegance for simplicity.
template <typename State, typename Action>
class MDPSolver {
   protected:
    Policy<State, Action> m_target_policy;                      // Target policy π
    Policy<State, Action> m_behavior_policy;                    // Behavioral policy β
    std::unordered_map<State, Return, StateHash<State>> m_v{};  // State-value v function
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q;  // Action-value Q function

    MDP<State, Action> *m_mdp;

    virtual void initialize_policies() {
        for (const State &s : this->m_mdp->S()) {
            m_target_policy.set(s, 0);    // Initialize target policy
            m_behavior_policy.set(s, 0);  // Initialize behavioral policy
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

    explicit MDPSolver(MDP<State, Action> *mdp_core) : m_mdp(mdp_core) {}

    virtual void initialize() {
        initialize_policies();
        initialize_value_functions();
    }

    void set_v(State s, Return r) { m_v[s] = r; }
    void set_q(State s, Action a, Return r) { m_Q[{s, a}] = r; }
    void set_target_policy(State s, Action a) { m_target_policy[s] = a; }
    void set_behavior_policy(State s, Action a) { m_behavior_policy[s] = a; }

    std::unordered_map<State, Return, StateHash<State>> get_v() { return m_v; }

    Action target_policy(State s) { return m_target_policy(s); }
    Action behavior_policy(State s) { return m_behavior_policy(s); }

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

    Policy<State, Action> get_target_policy() { return m_target_policy; }
    Policy<State, Action> get_behavior_policy() { return m_behavior_policy; }

    std::vector<std::tuple<State, Action, Reward, State>> generate_episode() {
        std::vector<std::tuple<State, Action, Reward, State>> episode;
        State state = m_mdp->reset();
        bool done = false;

        while (!done) {
            Action action = m_behavior_policy(state);
            auto [next_state, reward] = m_mdp->step(action);
            episode.emplace_back(state, action, reward, next_state);
            state = next_state;
            done = m_mdp->is_terminal(state);
        }

        return episode;
    }
};
