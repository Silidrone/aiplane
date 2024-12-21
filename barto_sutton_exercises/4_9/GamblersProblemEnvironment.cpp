#include "GamblersProblemEnvironment.h"

#include <matplot/matplot.h>

#include "MDPSolver.h"
#include "m_utils.h"

void GamblersProblemEnvironment::initialize() {
    for (State s = MIN_STAKE + 1; s <= MAX_STAKE - 1; s++) {
        m_S.emplace_back(s);

        for (Action a = MIN_STAKE; a <= std::min(s, MAX_STAKE - s); a++) {
            m_A[s].emplace_back(a);
            for (auto& coin_flip : coin_flips) {
                if (coin_flip == HEADS) {
                    State next_state = s + a;
                    Reward reward = (next_state == MAX_STAKE) ? 1 : 0;

                    m_dynamics[{s, a}].emplace_back(next_state, reward, P_H);
                } else {
                    State next_state = s - a;
                    Reward reward = (next_state == MIN_STAKE) ? 0 : 0;

                    m_dynamics[{s, a}].emplace_back(next_state, reward, P_T);
                }
            }
        }
    }
}

void GamblersProblemEnvironment::plot_policy(Policy<State, Action>& pi) {
    matplot::vector_1d x, y;

    // Collecting states (amount of money)
    for (int i = MIN_STAKE + 1; i < MAX_STAKE; ++i) {
        x.push_back(i);
    }

    // Collecting actions (bet sizes)
    for (int i = MIN_STAKE + 1; i < MAX_STAKE; ++i) {
        y.push_back(static_cast<double>(pi(i)));
    }

    // Plotting the policy
    matplot::plot(x, y);

    // Setting axis labels
    matplot::xlabel("Amount of money (state)");
    matplot::ylabel("Bet size (action)");
    matplot::title("Policy Plot for Gambler's Problem");

    // Adjusting x and y ticks to show all intermediate values
    matplot::xticks(x);  // Automatically sets ticks to match the x data values
    matplot::yticks(y);  // Automatically sets ticks to match the y data values

    // Display the plot
    matplot::show();
}