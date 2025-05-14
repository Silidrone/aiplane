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

    // Optional experience replay
    bool use_replay_buffer;
    ReplayBuffer<State, Action> replay_buffer;
    int batch_size;

    // Whether we decay epsilon
    bool decay_epsilon;

   public:
    SARSA(MDP<State, Action>* mdp_core, Policy<State, Action>* policy, ValueStrategy<State, Action>* value_strategy,
          const double discount_rate, const long double policy_threshold, bool decay_eps = false,
          bool use_experience_replay = false, int buffer_capacity = 10000, int batch = 32)
        : GPI<State, Action>(mdp_core, policy, discount_rate, policy_threshold),
          m_value_strategy(value_strategy),
          decay_epsilon(decay_eps),
          use_replay_buffer(use_experience_replay),
          replay_buffer(buffer_capacity),
          batch_size(batch) {
        policy->initialize(mdp_core, value_strategy);
    };

    void update_from_batch() {
        if (replay_buffer.size() < batch_size) {
            return;
        }

        auto batch = replay_buffer.sample(batch_size);

        for (const auto& experience : batch) {
            const auto& [state, action, reward, next_state, done] = experience;

            double target_q;
            if (done) {
                target_q = reward;
            } else {
                Action next_action = this->m_policy->sample(next_state);
                double q_next = m_value_strategy->Q(next_state, next_action);
                target_q = reward + this->m_discount_rate * q_next;
            }

            m_value_strategy->update(state, action, target_q);
        }
    }

    void sarsa_main() {
        int episode = 0;
        auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy<State, Action>*>(this->m_policy);

        std::cout << "Starting SARSA training..." << std::endl;

        do {  // episode loop
            if (episode % 10 == 0) {
                std::cout << "Episode " << episode;
                if (eps_policy) {
                    std::cout << " - Epsilon: " << eps_policy->get_epsilon();
                }
                std::cout << std::endl;
            }

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

                if (use_replay_buffer) {
                    replay_buffer.add(s, a, r, s_prime, done);
                    update_from_batch();
                } else {
                    double target_q;
                    if (done) {
                        target_q = r;
                    } else {
                        double q_next = m_value_strategy->Q(s_prime, a_prime);
                        target_q = r + this->m_discount_rate * q_next;
                    }

                    m_value_strategy->update(s, a, target_q);
                }

                s = s_prime;
                a = a_prime;

            } while (!this->m_mdp->is_terminal(s));

            if (decay_epsilon) {
                auto* eps_policy = dynamic_cast<EpsilonGreedyPolicy<State, Action>*>(this->m_policy);
                if (eps_policy) {
                    eps_policy->decay_epsilon();
                }
            };

            if (episode % 10 == 0) {
                std::cout << "Episode " << episode << " - Steps: " << steps << " - Reward: " << episode_reward
                          << std::endl;
            }

        } while (episode < this->m_policy_threshold);

        std::cout << "Training completed after " << episode << " episodes" << std::endl;
    }

    void policy_iteration() override { sarsa_main(); }
};