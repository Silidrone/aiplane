#pragma once

#include <limits>

#include "MDPSolver.h"

template <typename State, typename Action>
class ValueIteration : public MDPSolver<State, Action> {
   protected:
    double m_discount_rate;
    double m_policy_threshold;

    void update_final_policy();

   public:
    ValueIteration(MDPCore<State, Action>*, const double, const double);

    void value_iteration();
};
