#include "QValuePolicyIteration.h"

template <typename State, typename Action>
QValuePolicyIteration<State, Action>::QValuePolicyIteration(MDPCore<State, Action> *mdp_core)
    : PolicyIteration<State, Action>(mdp_core, 0.9f, 0.01f) {}

template <typename State, typename Action>
void QValuePolicyIteration<State, Action>::policy_evaluation() {
    Return delta;
    do {
        delta = 0;
        for (State &s : this->m_mdp->S()) {
            for (Action &a : this->m_mdp->A(s)) {
                const Return old_value = this->Q(s, a);
                Return new_value = 0;
                auto transitions = this->m_mdp->p(s, a);

                for (auto transition : transitions) {
                    State s_prime = std::get<0>(transition);
                    Reward r = std::get<1>(transition);
                    double probability = std::get<2>(transition);
                    new_value += probability * (r + this->m_discount_rate * this->Q(s_prime, this->pi(s_prime)));
                }

                this->m_Q[{s, a}] = new_value;
                delta = std::max(delta, std::abs(old_value - new_value));
            }
        }
    } while (delta > this->m_policy_evaluation_threshold);
}

template <typename State, typename Action>
bool QValuePolicyIteration<State, Action>::policy_improvement() {
    bool policy_stable = true;
    for (State &s : this->m_mdp->S()) {
        const Return old_value = this->Q(s, this->pi(s));
        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;
        for (Action &a : this->m_mdp->A(s)) {
            Return candidate_return = this->Q(s, a);
            if (candidate_return > max_return) {
                max_return = candidate_return;
                maximizing_action = a;
            }
        }

        this->m_pi.set(s, maximizing_action);

        if (old_value != max_return) {
            policy_stable = false;
        }
    }

    return policy_stable;
}

template class QValuePolicyIteration<std::vector<int>, int>;
template class QValuePolicyIteration<int, int>;
