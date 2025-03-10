#include "QValuePolicyIteration.h"

template <typename State, typename Action>
QValuePolicyIteration<State, Action>::QValuePolicyIteration(MDP<State, Action> &mdp_core, const double discount_rate,
                                                            const long double policy_threshold)
    : GPI<State, Action>(mdp_core, discount_rate, policy_threshold) {}

template <typename State, typename Action>
void QValuePolicyIteration<State, Action>::policy_evaluation() {
    Return delta;
    do {
        delta = 0;
        for (State &s : this->m_mdp.S()) {
            for (Action &a : this->m_mdp.A(s)) {
                const Return old_value = this->Q(s, a);
                Return new_value = 0;
                auto transitions = this->m_mdp.p(s, a);

                for (auto transition : transitions) {
                    State s_prime = std::get<0>(transition);
                    Reward r = std::get<1>(transition);
                    double probability = std::get<2>(transition);
                    new_value +=
                        probability * (r + this->m_discount_rate * this->Q(s_prime, this->m_policy.sample(s_prime)));
                }

                this->m_Q[{s, a}] = new_value;
                delta = std::max(delta, std::abs(old_value - new_value));
            }
        }
    } while (delta > this->m_policy_threshold);
}

template <typename State, typename Action>
bool QValuePolicyIteration<State, Action>::policy_improvement() {
    bool policy_stable = true;
    for (State &s : this->m_mdp.S()) {
        const Return old_value = this->Q(s, this->m_policy.sample(s));
        auto [maximizing_action, max_return] = this->Q_best_action(s);

        this->m_policy.set(s, maximizing_action);

        if (old_value != max_return) {
            policy_stable = false;
        }
    }

    return policy_stable;
}

template class QValuePolicyIteration<std::vector<int>, int>;
template class QValuePolicyIteration<int, int>;
