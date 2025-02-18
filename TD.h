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
       const long double policy_threshold, const double step_size)
        : GPI<State, Action>(mdp_core, discount_rate, policy_threshold), m_policy(policy), step_size(step_size) {
        m_policy->initialize(this);
    };

    void td_main() {
        int i = 0;
        do {  // episode loop
            i++;
            State s = this->m_mdp.reset();
            Action a = this->m_policy->sample(s);
            do {  // step loop
                auto [s_prime, r] = this->m_mdp.step(s, a);
                if (this->m_mdp.is_terminal(s_prime)) {
                    this->m_Q[{s, a}] = this->Q(s, a) + this->step_size * (r - this->Q(s, a));
                } else {
                    Action a_prime = this->m_policy->sample(s_prime);
                    this->m_Q[{s, a}] =
                        this->Q(s, a) +
                        this->step_size * (r + this->m_discount_rate * this->Q(s_prime, a_prime) - this->Q(s, a));
                    a = a_prime;
                }

                s = s_prime;
            } while (!this->m_mdp.is_terminal(s));
        } while (i < this->m_policy_threshold);  // m_policy_threshold represents the # of episodes before termination
    }

    void policy_iteration() override { td_main(); }
};