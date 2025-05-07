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
            // if ((i % 1000) == 0) {
            //     std::cout << "Episode " << i << " - Testing performance:" << std::endl;

            //     // Run 5 test episodes with greedy policy (epsilon=0)
            //     double total_survival_time = 0;
            //     const int NUM_TEST_EPISODES = 5;
            //     double total_reward = 0.0;

            //     for (int test_ep = 0; test_ep < NUM_TEST_EPISODES; test_ep++) {
            //         State s = this->m_mdp->reset();
            //         int steps = 0;
            //         double episode_reward = 0.0;

            //         while (!this->m_mdp->is_terminal(s) && steps < 1000) {
            //             // Use greedy action selection - get action from greedy_action tuple
            //             Action a = std::get<0>(this->m_policy->greedy_action(s));
            //             auto [s_prime, r] = this->m_mdp->step(s, a);
            //             episode_reward += r;
            //             s = s_prime;
            //             steps++;
            //         }

            //         total_survival_time += steps;
            //         total_reward += episode_reward;
            //         std::cout << "  Episode " << test_ep << ": survived " << steps
            //                   << " steps, reward: " << episode_reward << std::endl;
            //     }

            //     double avg_survival = total_survival_time / NUM_TEST_EPISODES;
            //     double avg_reward = total_reward / NUM_TEST_EPISODES;
            //     std::cout << "Average survival time: " << avg_survival << " steps, Average reward: " << avg_reward
            //               << std::endl;
            //     std::cout << "------------------------------" << std::endl;

            //     // Display top feature weights (useful for understanding what the network learns)
            //     const auto& weights = m_value_strategy->get_approximator()->get_weights();
            //     std::cout << "Feature weights sample (first 10 weights):" << std::endl;
            //     for (size_t j = 0; j < std::min(size_t(10), weights.size()); ++j) {
            //         std::cout << "  Weight[" << j << "] = " << weights[j] << std::endl;
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