#pragma once

#include <random>

#include "DerivedPolicy.h"

template <typename State, typename Action>
class DerivedStochasticPolicy : public DerivedPolicy<State, Action> {
   private:
    double m_epsilon;
    std::mt19937 m_generator;

   public:
    DerivedStochasticPolicy(double epsilon) : m_epsilon(epsilon), m_generator(std::random_device{}()) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
    }

    Action sample(const State& s) override {
        if (this->m_mdp_solver == nullptr) {
            throw std::logic_error("initialize must be called first!");
        }

        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        if (distribution(m_generator) < m_epsilon) {
            return this->m_mdp_solver->random_action(s);
        } else {
            auto [best_action, _] = this->m_mdp_solver->Q_best_action(s);
            return best_action;
        }
    }
};
