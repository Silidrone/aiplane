#include "CarRentalEnvironment.h"

#include <matplot/matplot.h>

#include "m_utils.h"
#include "matplot/util/colors.h"

void CarRentalEnvironment::generate_requests_returns_combinations(const State& s) {
    std::vector<std::pair<int, int>> loc1_rr;
    for (int i = 0; i <= 4; i++) {
        for (int j = 0; j <= 4; j++) {
            loc1_rr.emplace_back(i, j);
        }
    }

    std::vector<std::pair<int, int>> loc2_rr;
    for (int i = 0; i <= 5; i++) {
        for (int j = 0; j <= 3; j++) {
            loc2_rr.emplace_back(i, j);
        }
    }

    for (const auto& rr1 : loc1_rr) {
        for (const auto& rr2 : loc2_rr) {
            m_rrs[s].push_back({rr1, rr2});
        }
    }
}

void CarRentalEnvironment::generate_dynamics_p() {
    for (const State& s : m_S) {
        for (Action a : this->A(s)) {
            auto state_rrs = m_rrs[s];

            for (auto& state_rr : state_rrs) {
                State s_prime = s;
                Reward reward = 0;

                // move the cars
                reward -= abs(a * COST_OF_MOVING_A_CAR);
                s_prime[0] -= a;
                s_prime[1] += a;

                double probability = 1;
                bool invalid_state = false;
                for (int loc = 0; loc < s.size(); loc++) {
                    auto [requested, returned] = state_rr[loc];
                    probability *= poisson_probability(requested, requests_lambda[loc]) *
                                   poisson_probability(returned, returns_lambda[loc]);

                    // rent out the cars
                    const int fulfilled_requests = std::min(requested, s_prime[loc]);
                    reward += fulfilled_requests * REQUEST_FULFILLMENT_REWARD;
                    s_prime[loc] -= fulfilled_requests;

                    // return the cars
                    s_prime[loc] += returned;

                    if (s_prime[loc] > MAX_CARS_COUNT_PER_LOCATION) {
                        invalid_state = true;
                        break;
                    }
                }

                if (!invalid_state) {
                    m_dynamics[{s, a}].emplace_back(s_prime, reward, probability);
                }
            }
        }
    }
}

void CarRentalEnvironment::initialize() {
    for (int l1 = 0; l1 <= MAX_CARS_COUNT_PER_LOCATION; l1++) {
        for (int l2 = 0; l2 <= MAX_CARS_COUNT_PER_LOCATION; l2++) {
            State s = {l1, l2};
            m_S.push_back(s);
            for (int a = -1 * std::min(std::min(4, l2), MAX_CARS_COUNT_PER_LOCATION - l1);
                 a <= std::min(std::min(5, l1), MAX_CARS_COUNT_PER_LOCATION - l2); a++) {
                m_A[s].emplace_back(a);
            }
            generate_requests_returns_combinations(s);
        }
    }

    generate_dynamics_p();
}
void CarRentalEnvironment::plot_policy(Policy<State, Action>& pi) {
    int grid_size_x = 21;
    int grid_size_y = 21;

    matplot::vector_1d x, y;
    matplot::vector_2d Z;

    Z.resize(grid_size_x, std::vector<double>(grid_size_y, 0));

    for (int i = 0; i < grid_size_x; ++i) {
        x.push_back(i);
    }

    for (int j = 0; j < grid_size_y; ++j) {
        y.push_back(j);
    }

    auto [X, Y] = matplot::meshgrid(x, y);

    for (auto& p : pi.map_container()) {
        auto s = p.first;
        Z[s[0]][s[1]] = static_cast<double>(pi(s));
    }

    auto contour = matplot::contour(X, Y, Z);
    contour->color("black");
    contour->line_width(2);

    matplot::xlabel("Cars at second location");
    matplot::ylabel("Cars at first location");
    matplot::title("Policy Contour");

    matplot::show();
}
