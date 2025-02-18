#pragma once
#include <MDPSolver.h>

template <typename State, typename Action>
class Policy {
   public:
    virtual void initialize(MDPSolver<State, Action>*) = 0;
    virtual Action sample(const State&) = 0;
};
