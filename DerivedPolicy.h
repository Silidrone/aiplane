#pragma once

#include <stdexcept>

#include "MDPSolver.h"
#include "m_utils.h"

template <typename State, typename Action>
class DerivedPolicy : public Policy<State, Action> {
   protected:
    MDPSolver<State, Action>* m_mdp_solver;

   public:
    DerivedPolicy() : m_mdp_solver(nullptr) {}
    virtual ~DerivedPolicy() = default;

    void initialize(MDPSolver<State, Action>* mdp_solver) override { m_mdp_solver = mdp_solver; }

    virtual Action sample(const State& s) override = 0;
};
