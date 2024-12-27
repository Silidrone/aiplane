#pragma once
#include <limits>

#include "GPI.h"

template <typename State, typename Action>
class VValuePolicyIteration : public GPI<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;

   public:
    explicit VValuePolicyIteration(MDPCore<State, Action>*, const double, const double);
};
