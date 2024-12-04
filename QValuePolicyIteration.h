#ifndef QVALUEPOLICYITERATION_H
#define QVALUEPOLICYITERATION_H

#include "PolicyIteration.h"
#include <limits>

template <typename State, typename Action>
class QValuePolicyIteration : public PolicyIteration<State, Action> {
protected:
    void initialize_policy() override {
        for (const State &s : this->m_mdp->S())
        {
            this->m_pi.set(s, 0);
        }
    }

    void initialize_value_functions() override {
        for (const State &s : this->m_mdp->S())
        {
            this->m_v[s] = 0;
            for (const Action &a : this->m_mdp->A()) {
                this->m_Q[{s, a}] = 0;
            }
        }
    }

    void policy_evaluation() override
    {
        Return delta;
        do
        {
            delta = 0;
            for (State &s : this->m_mdp->S())
            {
                for (Action &a : this->m_mdp->A())
                {
                    const Return old_value = q_f(s, a);
                    Return new_value = 0;
                    auto transitions = this->m_mdp->p(s, a);

                    for (auto transition : transitions)
                    {
                        State s_prime = std::get<0>(transition);
                        Reward r = std::get<1>(transition);
                        double probability = std::get<2>(transition);
                        new_value += probability * (r + this->m_discount_rate * q_f(s_prime, pi_f(s_prime)));
                    }

                    this->m_Q[{s, a}] = new_value;
                    delta = std::max(delta, std::abs(old_value - new_value));
                }
            }
        } while (delta > this->m_policy_evaluation_threshold);
    }

    bool policy_improvement() override
    {
        bool policy_stable = true;
        for (State &s : this->m_mdp->S())
        {
            const Return old_value = q_f(s, pi_f(s));
            Return max_return = std::numeric_limits<Return>::lowest(); // in case we decide to use negative rewards in the future
            Action maximizing_action;
            for (Action &a : this->m_mdp->A())
            {
                Return candidate_return = q_f(s, a);
                if (candidate_return > max_return)
                {
                    max_return = candidate_return;
                    maximizing_action = a;
                }
            }

            // maximizing_action should, theoretically, always be initialized
            this->m_pi.set(s, maximizing_action);

            if (old_value != max_return)
            {
                policy_stable = false;
            }
        }

        return policy_stable;
    }
public:
explicit QValuePolicyIteration(MDPCore<State, Action>* mdp_core)
    : PolicyIteration<State, Action>(mdp_core, 0.9f, 0.01f) {}

};

#endif //QVALUEPOLICYITERATION_H
