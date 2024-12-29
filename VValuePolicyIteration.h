#pragma once
#include <limits>

#include "DeterministicPolicy.h"
#include "GPI.h"

template <typename State, typename Action>
class VValuePolicyIteration : public GPI<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;
    DeterministicPolicy<State, Action> m_policy;
    void initialize() override;

   public:
    explicit VValuePolicyIteration(MDP<State, Action>&, const double, const double);
    DeterministicPolicy<State, Action> get_policy() const { return m_policy; }
};
