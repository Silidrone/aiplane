#pragma once

#include <limits>

#include "PolicyIteration.h"

template <typename State, typename Action>
class QValuePolicyIteration : public PolicyIteration<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;

   public:
    explicit QValuePolicyIteration(MDPCore<State, Action>*, const double, const double);
};
