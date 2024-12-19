#pragma once

#include <limits>
#include "MDPSolver.h"

template <typename State, typename Action>
class ValueIteration : public MDPSolver<State, Action> {
protected:
    double DISCOUNT_RATE = 0.9f;
    double VALUE_ITERATION_THRESHOLD_EPSILON = 0.01f;
    void update_final_policy();
public:
    ValueIteration(MDPCore<State, Action>* mdp_core);

    void policy_iteration();
};
