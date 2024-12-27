#pragma once

#include <limits>

#include "GPI.h"

template <typename State, typename Action>
class QValuePolicyIteration : public GPI<State, Action> {
   protected:
    void policy_evaluation() override;
    bool policy_improvement() override;

   public:
    explicit QValuePolicyIteration(MDPCore<State, Action>*, const double, const double);
};
