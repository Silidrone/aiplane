#pragma once

#include <limits>
#include "MDPSolver.h"

template <typename State, typename Action>
class ValueIteration : public MDPSolver<State, Action> {
protected:
    double VALUE_ITERATION_THRESHOLD_EPSILON = 0.01f;
    double m_discount_rate;
    double m_policy_evaluation_threshold;

public:
    ValueIteration(MDPCore<State, Action>* mdp_core, const double discount_rate, const double policy_evaluation_threshold);

    void policy_iteration();
};
