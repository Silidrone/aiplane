#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DeterministicPolicy.h"
#include "ESoftPolicy.h"
#include "MC_FV.h"
#include "barto_sutton_exercises/5_1/Blackjack.h"

static constexpr int N_OF_EPISODES = 100000;

inline void plot_v_f(MDPSolver<State, Action>& mdp_solver) {
    matplot::vector_2d x, y, z_usable, z_no_usable;

    for (int player_sum = MIN_PLAYER_SUM; player_sum < MAX_SUM; ++player_sum) {
        matplot::vector_1d x_row, y_row, z_row_usable, z_row_no_usable;

        for (int dealer_face_up = 1; dealer_face_up <= 10; ++dealer_face_up) {
            x_row.push_back(player_sum);
            y_row.push_back(dealer_face_up);

            State state_with_usable = {player_sum, dealer_face_up, true};
            State state_without_usable = {player_sum, dealer_face_up, false};

            z_row_usable.push_back(mdp_solver.v(state_with_usable));
            z_row_no_usable.push_back(mdp_solver.v(state_without_usable));
        }

        x.push_back(x_row);
        y.push_back(y_row);
        z_usable.push_back(z_row_usable);
        z_no_usable.push_back(z_row_no_usable);
    }

    // You have to uncomment either one if you want to see it and comment the other, they can't be shown (to my
    // knowledge) both at the same time.

    // Plot for usable ace
    // auto fig1 = matplot::figure();
    // matplot::surf(x, y, z_usable);
    // matplot::xlabel("Player's Total Sum");
    // matplot::ylabel("Dealer's Face-Up Card");
    // matplot::zlabel("State-Value V(s)");
    // matplot::title("V-Function with Usable Ace");
    // matplot::zlim({LOSS_REWARD, WIN_REWARD});
    // matplot::show();

    // Plot for no usable ace
    auto fig2 = matplot::figure();
    matplot::surf(x, y, z_no_usable);
    matplot::xlabel("Player's Total Sum");
    matplot::ylabel("Dealer's Face-Up Card");
    matplot::zlabel("State-Value V(s)");
    matplot::title("V-Function without Usable Ace");
    matplot::zlim({LOSS_REWARD, WIN_REWARD});
    matplot::show();
}

inline int blackjack_main() {
    Blackjack environment;
    environment.initialize();

    ESoftPolicy<State, Action> policy(0.1);
    policy.initialize(environment.S(), environment.A());

    MC_FV<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES);
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    serialize_to_json(mdp_solver.get_v(), "blackjack-value-function-estimation.json");
    plot_v_f(mdp_solver);

    return 0;
}
