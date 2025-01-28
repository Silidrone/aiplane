#pragma once

#include "MDPSolver.h"

template <typename State, typename Action>
class GPI : public MDPSolver<State, Action> {
   protected:
    double m_discount_rate;
    long double m_policy_threshold;

    virtual void policy_evaluation(){};
    virtual bool policy_improvement() { return false; };

   public:
    GPI(MDP<State, Action>& mdp_core, const double discount_rate, const long double policy_threshold)
        : MDPSolver<State, Action>(mdp_core), m_discount_rate(discount_rate), m_policy_threshold(policy_threshold) {}

    virtual void policy_iteration() {
        bool policy_stable;
        do {
            policy_evaluation();
            policy_stable = policy_improvement();
        } while (!policy_stable);
    }
};
