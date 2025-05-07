#pragma once

#include <chrono>
#include <iostream>
#include <limits>

#include "FunctionApproximator.h"
#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action>
class FA_TD : public GPI<State, Action> {
   protected:
    const double step_size;
    ApproximationValueStrategy<State, Action>* m_value_strategy;

   public:
    FA_TD(MDP<State, Action>* mdp_core, Policy<State, Action>* policy,
          ApproximationValueStrategy<State, Action>* value_strategy, const double discount_rate,
          const long double policy_threshold, const double step_size)
        : GPI<State, Action>(mdp_core, policy, discount_rate, policy_threshold),
          m_value_strategy(value_strategy),
          step_size(step_size) {
        policy->initialize(mdp_core, value_strategy);
    };

    void td_main() {
        int i = 0;
        do {  // episode loop
            // if ((i % 100) == 0) {
            //     std::cout << "episodes finished " << i << std::endl;
            //     std::cout << "Episode " << i << " - Current weights and their features:" << std::endl;
            //     const auto& weights = m_value_strategy->get_approximator()->get_weights();
            //     std::vector<std::string> feature_descriptions = {
            //         "MyPosX: Normalized X position of agent",
            //         "MyPosY: Normalized Y position of agent",
            //         "MyVelX: Normalized X velocity of agent",
            //         "MyVelY: Normalized Y velocity of agent",
            //         "TagPosX: Normalized X position of tagger",
            //         "TagPosY: Normalized Y position of tagger",
            //         "TagVelX: Normalized X velocity of tagger",
            //         "TagVelY: Normalized Y velocity of tagger",
            //         "MyPosX×ActionX: Position-action cross feature",
            //         "MyPosY×ActionY: Position-action cross feature",
            //         "MyVelX×ActionX: Velocity-action cross feature",
            //         "MyVelY×ActionY: Velocity-action cross feature",
            //         "TagPosX×ActionX: Tagger position-action cross feature",
            //         "TagPosY×ActionY: Tagger position-action cross feature",
            //         "TagVelX×ActionX: Tagger velocity-action cross feature",
            //         "TagVelY×ActionY: Tagger velocity-action cross feature",
            //         "Bias: Constant term"};

            //     for (size_t j = 0; j < weights.size(); ++j) {
            //         std::cout << "  Weight[" << j << "] = " << weights[j] << " | Feature: " <<
            //         feature_descriptions[j]
            //                   << std::endl;
            //     }
            //     std::cout << "------------------------------" << std::endl;
            // }
            i++;
            State s = this->m_mdp->reset();
            Action a = this->m_policy->sample(s);
            do {  // step loop
                auto [s_prime, r] = this->m_mdp->step(s, a);
                double q_current = m_value_strategy->get_approximator()->predict(s, a);

                if (this->m_mdp->is_terminal(s_prime)) {
                    double error = r - q_current;
                    m_value_strategy->get_approximator()->update(s, a, error, this->step_size);
                } else {
                    Action a_prime = this->m_policy->sample(s_prime);
                    double q_next = m_value_strategy->get_approximator()->predict(s_prime, a_prime);
                    double error = (r + this->m_discount_rate * q_next) - q_current;
                    m_value_strategy->get_approximator()->update(s, a, error, this->step_size);
                    a = a_prime;
                }

                s = s_prime;
            } while (!this->m_mdp->is_terminal(s));
        } while (i < this->m_policy_threshold);
    }

    void policy_iteration() override { td_main(); }
};