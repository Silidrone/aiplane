#include "ValueIteration.h"

template <typename State, typename Action>
ValueIteration<State, Action>::ValueIteration(MDPCore<State, Action> *mdp_core) : MDPSolver<State, Action>(mdp_core) {}

template <typename State, typename Action>
void ValueIteration<State, Action>::policy_iteration() {
    Return delta;
    bool policy_stable;
    do {
        delta = 0;
        for (State &s : this->m_mdp->S()) {
            Return old_value = this->v(s);
            Return max_value = std::numeric_limits<Return>::lowest();
            for (Action &a : this->m_mdp->A(s)) {
                Return new_value = 0;
                auto transitions = this->m_mdp->p(s, a);

                for (auto transition : transitions) {
                    State s_prime = std::get<0>(transition);
                    Reward r = std::get<1>(transition);
                    double probability = std::get<2>(transition);
                    new_value += probability * (r + DISCOUNT_RATE * this->v(s_prime));
                }

                if (new_value > max_value) {
                    max_value = new_value;
                }
            }
            this->m_v[s] = max_value;
            delta = std::max(delta, std::abs(old_value - max_value));
        }
    } while (delta > VALUE_ITERATION_THRESHOLD_EPSILON);

    update_final_policy();
}

template <typename State, typename Action>
void ValueIteration<State, Action>::update_final_policy() {
    for (State &s : this->m_mdp->S()) {
        Return old_value = this->v(s);
        Action max_action;
        Return max_value = std::numeric_limits<Return>::lowest();
        for (Action &a : this->m_mdp->A(s)) {
            Return new_value = 0;
            auto transitions = this->m_mdp->p(s, a);

            for (auto transition : transitions) {
                State s_prime = std::get<0>(transition);
                Reward r = std::get<1>(transition);
                double probability = std::get<2>(transition);
                new_value += probability * (r + DISCOUNT_RATE * this->v(s_prime));
            }

            if (new_value > max_value) {
                max_value = new_value;
                max_action = a;
            }
        }
        this->m_pi[s] = max_action;
    }
}

template class ValueIteration<std::vector<int>, int>;
template class ValueIteration<int, int>;
