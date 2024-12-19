#pragma once
#include <limits>
#include "PolicyIteration.h"

template <typename State, typename Action>
class VValuePolicyIteration : public PolicyIteration<State, Action> {
protected:
    static constexpr double DISCOUNT_RATE = 0.9f;
    static constexpr double POLICY_THRESHOLD_EPSILON = 0.01f;

    void policy_evaluation() override;
    bool policy_improvement() override;

public:
    explicit VValuePolicyIteration(MDPCore<State, Action>* mdp_core);
};
