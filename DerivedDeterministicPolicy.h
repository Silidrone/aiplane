#pragma once

#include "DerivedPolicy.h"

template <typename State, typename Action>
class DerivedDeterministicPolicy : public DerivedPolicy<State, Action> {
   public:
    DerivedDeterministicPolicy() = default;

    Action sample(const State& s) override {
        if (this->m_mdp_solver == nullptr) {
            throw std::logic_error("initialize must be called first!");
        }

        auto [best_action, _] = this->m_mdp_solver->Q_best_action(s);
        return best_action;
    }
};
