#pragma once

#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "GPI.h"
#include "Policy.h"
#include "ReplayBuffer.h"
#include "ValueStrategy.h"
#include "m_utils.h"

template <typename State, typename Action>
class SARSA : public GPI<State, Action> {
   protected:
    ValueStrategy<State, Action>* m_value_strategy;

    // Whether we decay epsilon
    bool decay_epsilon;

    int print_freq = 100;  // Default: time every 10th episode

    // Wrapper to time the update operation
    void timed_update(const State& s, const Action& a, double target_q, int current_episode = 0) {
        if (current_episode % print_freq == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            m_value_strategy->update(s, a, target_q);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Episode " << current_episode << " - Update time: " << duration.count() << " ms" << std::endl;
        } else {
            m_value_strategy->update(s, a, target_q);
        }
    }

    // Wrapper to time the Q operation
    double timed_Q(const State& s, const Action& a, int current_episode = 0) {
        if (current_episode % print_freq == 0) {
            auto start = std::chrono::high_resolution_clock::now();
            double q_value = m_value_strategy->Q(s, a);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            std::cout << "Episode " << current_episode << " - Q lookup time: " << duration.count() << " ms"
                      << std::endl;
            return q_value;
        } else {
            return m_value_strategy->Q(s, a);
        }
    }

   public:
    SARSA(MDP<State, Action>* mdp_core, Policy<State, Action>* policy, ValueStrategy<State, Action>* value_strategy,
          const double discount_rate, const long double policy_threshold, bool decay_eps = false)
        : GPI<State, Action>(mdp_core, policy, discount_rate, policy_threshold),
          m_value_strategy(value_strategy),
          decay_epsilon(decay_eps) {
        policy->initialize(mdp_core, value_strategy);
    };

    void sarsa_main() {
        int episode = 0;
        auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy<State, Action>*>(this->m_policy);

        std::cout << "Starting SARSA training..." << std::endl;

        do {  // episode loop
            episode++;
            State s = this->m_mdp->reset();
            Action a = this->m_policy->sample(s);
            double episode_reward = 0;
            int steps = 0;

            do {  // step loop
                steps++;
                auto [s_prime, r] = this->m_mdp->step(s, a);
                episode_reward += r;

                bool done = this->m_mdp->is_terminal(s_prime);
                Action a_prime = done ? a : this->m_policy->sample(s_prime);

                double target_q;
                if (done) {
                    target_q = r;
                } else {
                    //                    double q_next = timed_Q(s_prime, a_prime, episode);
                    double q_next = m_value_strategy->Q(s_prime, a_prime);
                    target_q = r + this->m_discount_rate * q_next;
                }

                // m_value_strategy->update(s, a, target_q, episode);
                m_value_strategy->update(s, a, target_q);

                s = s_prime;
                a = a_prime;

            } while (!this->m_mdp->is_terminal(s));

            if (decay_epsilon) {
                auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy<State, Action>*>(this->m_policy);
                if (eps_policy) {
                    eps_policy->decay_epsilon();
                }
            };

            if (episode % print_freq == 0) {
                std::cout << "Episode " << episode << " - Steps: " << steps << " - Reward: " << episode_reward
                          << std::endl;
            }

        } while (episode < this->m_policy_threshold);

        std::cout << "Training completed after " << episode << " episodes" << std::endl;
    }

    void policy_iteration() override { sarsa_main(); }
};