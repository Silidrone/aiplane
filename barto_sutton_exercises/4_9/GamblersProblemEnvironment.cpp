#include "GamblersProblemEnvironment.h"

#include "m_utils.h"

void GamblersProblemEnvironment::initialize() {
    for (State s = MIN_STAKE + 1; s <= MAX_STAKE - 1; s++) {
        m_S.emplace_back(s);

        for (auto& coin_flip : coin_flips) {
            for (Action a = MIN_STAKE; a <= std::min(s, MAX_STAKE - s); a++) {
                m_A[s].emplace_back(a);
                if (coin_flip == HEADS) {
                    State state = s + a;
                    Reward reward = (state == MAX_STAKE) ? 1 : 0;

                    m_dynamics[{s, a}].emplace_back(state, reward, P_H);
                } else {
                    State state = s - a;

                    m_dynamics[{s, a}].emplace_back(state, 0, P_T);
                }
            }
        }
    }
}
