#pragma once

#include "DeterministicPolicy.h"
#include "GPI.h"

template <typename State, typename Action>
class ValueIteration : public GPI<State, Action> {
   protected:
    void update_final_policy();
    DeterministicPolicy<State, Action> m_policy;

   public:
    ValueIteration(MDP<State, Action>&, const double, const double);

    void policy_iteration() override;
    DeterministicPolicy<State, Action> get_policy() const { return m_policy; }
};
