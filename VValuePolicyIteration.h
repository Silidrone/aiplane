#pragma once
#include <limits>

#include "DeterministicPolicy.h"
#include "GPI.h"

template <typename State, typename Action>
class VValuePolicyIteration : public GPI<State, Action> {
   protected:
    DeterministicPolicy<State, Action> m_policy;
    void policy_evaluation() override;
    bool policy_improvement() override;
    void initialize() override;

   public:
    explicit VValuePolicyIteration(MDP<State, Action>&, const double, const long double);
    DeterministicPolicy<State, Action> get_policy() const { return m_policy; }
};
