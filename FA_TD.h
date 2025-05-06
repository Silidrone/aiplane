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
            if ((i % 300) == 0) {  // every 300th episode
                std::cout << "Episode " << i << " - Current weights and their features:" << std::endl;
                const auto& weights = m_value_strategy->get_approximator()->get_weights();
                std::vector<std::string> feature_descriptions = {
                    "Dist×ActionX: +1=values moving right when far, -1=values moving left when far",
                    "Dist×ActionY: +1=values moving down when far, -1=values moving up when far",
                    "MovingAway: +1=values fleeing behavior, -1=values approaching predator",
                    "LeftWall×ActionX: +1=values wall awareness, -1=ignores left wall danger",
                    "RightWall×ActionX: +1=values wall awareness, -1=ignores right wall danger",
                    "TopWall×ActionY: +1=values wall awareness, -1=ignores top wall danger",
                    "BottomWall×ActionY: +1=values wall awareness, -1=ignores bottom wall danger",
                    "SpeedDiff×Action: +1=values speed advantage, -1=devalues speed advantage",
                    "PredatorAlignment: +1=values similar direction, -1=values opposite direction",
                    "Bias, Bias"};

                for (size_t j = 0; j < weights.size(); ++j) {
                    std::cout << "  Weight[" << j << "] = " << weights[j] << " | Feature: " <<
                    feature_descriptions[j]
                              << std::endl;
                }
                std::cout << "------------------------------" << std::endl;
            }
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