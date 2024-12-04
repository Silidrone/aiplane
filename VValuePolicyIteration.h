#ifndef VVALUEPOLICYITERATION_H
#define VVALUEPOLICYITERATION_H
#include "PolicyIteration.h"
#include <limits>


template <typename State, typename Action>
class VValuePolicyIteration : public PolicyIteration<State, Action> {
protected:
    static constexpr double DISCOUNT_RATE = 0.9f;
    static constexpr double POLICY_THRESHOLD_EPSILON = 0.01f;

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

    void policy_evaluation() override {
        Return delta;
        do
        {
            delta = 0;
            for (State &s : this->m_mdp->S())
            {
                Action a = this->pi(s);

                const Return old_value = this->v(s);
                Return new_value = 0;
                auto transitions = this->m_mdp->p(s, a);

                for (auto transition : transitions)
                {
                    State s_prime = std::get<0>(transition);
                    Reward r = std::get<1>(transition);
                    double probability = std::get<2>(transition);
                    new_value += probability * (r + DISCOUNT_RATE * this->v(s_prime));
                }

                this->m_v[s] = new_value;
                delta = std::max(delta, std::abs(old_value - new_value));
            }
        } while (delta > POLICY_THRESHOLD_EPSILON);
    }

    bool policy_improvement() override {
        bool policy_stable = true;
        for (State &s : this->m_mdp->S())
        {
            const Return old_value = this->v(s);
            Return max_value = std::numeric_limits<Return>::lowest(); // in case we decide to use negative rewards in the future
            Action maximizing_action;
            for (Action &a : this->m_mdp->A())
            {
                Return state_value = 0;
                auto transitions =  this->m_mdp->p(s, a);
                for (auto &transition : transitions) {
                    State s_prime = std::get<0>(transition);
                    Reward r = std::get<1>(transition);
                    double probability = std::get<2>(transition);
                    state_value += probability * (r + DISCOUNT_RATE * this->v(s_prime));
                }
                if (state_value > max_value)
                {
                    max_value = state_value;
                    maximizing_action = a;
                }
            }

            // maximizing_action should, theoretically, always be initialized
            this->m_pi.set(s, maximizing_action);

            if (old_value != max_value)
            {
                policy_stable = false;
            }
        }

        return policy_stable;
    }
public:
    explicit VValuePolicyIteration(MDPCore<State, Action>* mdp_core)
        : PolicyIteration<State, Action>(mdp_core, 0.9f, 0.01f) {}

};



#endif //VVALUEPOLICYITERATION_H
