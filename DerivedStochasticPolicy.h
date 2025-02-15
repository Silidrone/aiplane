#pragma once

#include <random>
#include <stdexcept>

#include "MDPSolver.h"

template <typename State, typename Action>
class DerivedStochasticPolicy {
   private:
    MDPSolver<State, Action>* m_mdp_solver;
    double m_epsilon;
    std::mt19937 m_generator;

   public:
    DerivedStochasticPolicy(double epsilon)
        : m_epsilon(epsilon), m_mdp_solver(nullptr), m_generator(std::random_device{}()) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
    }

    void initialize(MDPSolver<State, Action>* mdp_solver) { m_mdp_solver = mdp_solver; }

    Action epsilon_greedy_sample(const State& s) {
        if (m_mdp_solver == nullptr) {
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
