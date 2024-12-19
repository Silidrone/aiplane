#include "VValuePolicyIteration.h"

template <typename State, typename Action>
VValuePolicyIteration<State, Action>::VValuePolicyIteration(MDPCore<State, Action> *mdp_core)
    : PolicyIteration<State, Action>(mdp_core, DISCOUNT_RATE, POLICY_THRESHOLD_EPSILON) {}

template <typename State, typename Action>
void VValuePolicyIteration<State, Action>::policy_evaluation() {
    Return delta;
    do {
        delta = 0;
        for (State &s : this->m_mdp->S()) {
            Action a = this->pi(s);

            const Return old_value = this->v(s);
            Return new_value = 0;
            auto transitions = this->m_mdp->p(s, a);

            for (auto transition : transitions) {
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

template <typename State, typename Action>
bool VValuePolicyIteration<State, Action>::policy_improvement() {
    bool policy_stable = true;
    for (State &s : this->m_mdp->S()) {
        const Return old_value = this->v(s);
        Return max_value = std::numeric_limits<Return>::lowest();  // in case we decide to use
                                                                   // negative rewards
        Action maximizing_action;
        for (Action &a : this->m_mdp->A(s)) {
            Return state_value = 0;
            auto transitions = this->m_mdp->p(s, a);
            for (auto &transition : transitions) {
                State s_prime = std::get<0>(transition);
                Reward r = std::get<1>(transition);
                double probability = std::get<2>(transition);
                state_value += probability * (r + DISCOUNT_RATE * this->v(s_prime));
            }
            if (state_value > max_value) {
                max_value = state_value;
                maximizing_action = a;
            }
        }

        this->m_pi.set(s, maximizing_action);

        if (old_value != max_value) {
            policy_stable = false;
        }
    }

    return policy_stable;
}

template class VValuePolicyIteration<std::vector<int>, int>;
template class VValuePolicyIteration<int, int>;
