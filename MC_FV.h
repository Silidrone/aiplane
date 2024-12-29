#pragma once

#include "GPI.h"
#include "StochasticPolicy.h"
#include "m_utils.h"

template <typename State, typename Action>
class MC_FV : public GPI<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>> N;
    double m_epsilon;
    StochasticPolicy<State, Action> m_policy;

   public:
    MC_FV(MDP<State, Action>&, const double, const double, const double epsilon);

    void policy_iteration() override;
};