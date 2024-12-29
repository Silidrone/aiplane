#pragma once

#include <limits>

#include "DeterministicPolicy.h"
#include "GPI.h"

template <typename State, typename Action>
class QValuePolicyIteration : public GPI<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;
    DeterministicPolicy<State, Action> m_policy;

   public:
    explicit QValuePolicyIteration(MDP<State, Action>&, const double, const double);
    DeterministicPolicy<State, Action> get_policy() const { return m_policy; }
};
