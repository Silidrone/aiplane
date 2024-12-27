#pragma once

#include "GPI.h"

template <typename State, typename Action>
class ValueIteration : public GPI<State, Action> {
   protected:
    void update_final_policy();

   public:
    ValueIteration(MDP<State, Action>*, const double, const double);

    void policy_iteration() override;
};
