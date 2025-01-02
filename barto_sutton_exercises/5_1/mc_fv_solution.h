#include <matplot/matplot.h>

#include <chrono>
#include <exception>
#include <functional>
#include <iostream>

#include "DeterministicPolicy.h"
#include "ESoftPolicy.h"
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

inline void plot_policy(const MDP<State, Action>& mdp, const Policy<State, Action>& policy, bool usable_ace_flag) {
    std::vector<double> x, y;
    std::vector<double> colors;

    for (const auto& state : mdp.S()) {
        if (std::get<2>(state) == usable_ace_flag) {
            double player_sum = static_cast<double>(std::get<0>(state));
            double dealer_face_up = static_cast<double>(std::get<1>(state));
            bool action = policy.optimal_action(state);
            x.push_back(dealer_face_up);
            y.push_back(player_sum);
            colors.push_back(action ? 1.0 : 0.0);
        }
    }

    auto fig = matplot::figure(true);
    auto scatter_plot = matplot::scatter(x, y, 50, colors);
    scatter_plot->marker("x");
    matplot::colormap({{0.0, 0.0, 1.0}, {1.0, 0.0, 0.0}});
    matplot::xlabel("Dealer's Face-Up Card");
    matplot::ylabel("Player's Total Sum");
    matplot::title(usable_ace_flag ? "Policy with Usable Ace" : "Policy without Usable Ace");
    matplot::show();
}

inline int blackjack_main() {
    Blackjack environment;
    environment.initialize();

    ESoftPolicy<State, Action> policy(0.15);
    policy.initialize(environment.S(), environment.A());

    MC_FV<State, Action> mdp_solver(environment, &policy, DISCOUNT_RATE, N_OF_EPISODES);
    mdp_solver.initialize();

    double time_taken = benchmark([&]() { mdp_solver.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    DeterministicPolicy<State, Action> optimal_policy(policy);

    serialize_to_json(mdp_solver.get_Q(), "blackjack-optimal-Q.json");
    serialize_to_json(policy.get_container(), "blackjack-stochastic-optimal-policy.json");
    serialize_to_json(optimal_policy.get_container(), "blackjack-deterministic-optimal-policy.json");
    // plot_v_f(mdp_solver, true);
    plot_policy(environment, optimal_policy, true);

    return 0;
}
