#pragma once

#include "PolicyIteration.h"
#include <limits>

template <typename State, typename Action>
class QValuePolicyIteration : public PolicyIteration<State, Action> {
protected:
    // Declare methods that will be defined later
    void policy_evaluation() override;
    bool policy_improvement() override;

public:
    explicit QValuePolicyIteration(MDPCore<State, Action>* mdp_core);
};
