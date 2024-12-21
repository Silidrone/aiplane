#pragma once
#include <limits>

#include "PolicyIteration.h"

template <typename State, typename Action>
class VValuePolicyIteration : public PolicyIteration<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;

   public:
    explicit VValuePolicyIteration(MDPCore<State, Action>*, const double, const double);
};
