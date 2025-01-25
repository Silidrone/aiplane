#pragma once

#include <limits>

#include "GPI.h"
#include "Policy.h"
#include "m_utils.h"

template <typename State, typename Action>
class TD : public GPI<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, int, StateActionPairHash<State, Action>> N;
    Policy<State, Action>* m_policy;
    const double step_size;

   public:
    TD(MDP<State, Action>& mdp_core, Policy<State, Action>* policy, const double discount_rate,
       const double number_of_episodes, const double step_size)
        : GPI<State, Action>(mdp_core, discount_rate, number_of_episodes), m_policy(policy), step_size(step_size){};

    void td_main() {
        int i = 0;
        do {  // episode loop
            i++;
            State s = this->m_mdp.reset();
            Action a = this->m_policy->sample(s);
            do {  // step loop
                auto [s_prime, r] = this->m_mdp.step(s, a);
                if (this->m_mdp.is_terminal(s_prime)) {
                    this->m_Q[{s, a}] += this->step_size * (r - this->Q(s, a));
                } else {
                    Action a_prime = this->m_policy->sample(s_prime);
                    this->m_Q[{s, a}] +=
                        this->step_size * (r + this->m_discount_rate * this->Q(s_prime, a_prime) - this->Q(s, a));
                    a = a_prime;
                }

                auto [maximizing_action, max_return] = this->Q_best_action(s);
                this->m_policy->set(s, maximizing_action);
                s = s_prime;
            } while (!this->m_mdp.is_terminal(s));
        } while (i < this->m_policy_threshold);
    }

    void policy_iteration() override { td_main(); }
};