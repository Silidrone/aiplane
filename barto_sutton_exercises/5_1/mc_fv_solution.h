#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DerivedStochasticPolicy.h"
#include "MC_FV.h"
#include "barto_sutton_exercises/5_1/Blackjack.h"

static constexpr int N_OF_EPISODES = 500000;

inline void plot_v_f(MDPSolver<State, Action>& mdp_solver, bool usable_ace_flag) {
    matplot::vector_2d x, y, z;

    for (int player_sum = MIN_PLAYER_SUM; player_sum < MAX_SUM; ++player_sum) {
        matplot::vector_1d x_row, y_row, z_row;

        for (int dealer_face_up = ACE; dealer_face_up <= FACE_CARD; ++dealer_face_up) {
            x_row.push_back(player_sum);
            y_row.push_back(dealer_face_up);

            State state = {player_sum, dealer_face_up, usable_ace_flag};
            z_row.push_back(mdp_solver.v(state));
        }

        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    auto fig = matplot::figure();
    matplot::surf(x, y, z);
    matplot::xlabel("Player's Total Sum");
    matplot::ylabel("Dealer's Face-Up Card");
    matplot::zlabel("State-Value V(s)");
    matplot::title(usable_ace_flag ? "V-Function with Usable Ace" : "V-Function without Usable Ace");
    matplot::zlim({LOSS_REWARD, WIN_REWARD});
    matplot::show();
}

inline int blackjack_main() {
    Blackjack environment;
    environment.initialize();

    DerivedStochasticPolicy<State, Action> policy(0.15);

    MC_FV<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES);
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    auto optimal_policy = mdp_solver.get_optimal_policy();

    serialize_to_json(mdp_solver.get_Q(), "blackjack-optimal-Q.json");
    serialize_to_json(optimal_policy, "blackjack-optimal-policy.json");
    environment.plot_policy(optimal_policy, false);

    return 0;
}
