#include <matplot/matplot.h>

#include <chrono>
#include <functional>
#include <iostream>

#include "VValuePolicyIteration.h"
#include "ValueIteration.h"
#include "barto_sutton_exercises/4_7/CarRentalEnvironment.h"

template <typename State, typename Action>
void plot_policy(Policy<State, Action> &pi) {
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

    for (auto &p : pi.map_container()) {
        auto s = p.first;
        Z[s[0]][s[1]] = static_cast<double>(pi(s));
    }

    matplot::contour(X, Y, Z);

    matplot::xlabel("Cars at second location");
    matplot::ylabel("Cars at first location");
    matplot::title("Policy Contour");

    matplot::show();
}

int main() {
    CarRentalEnvironment car_rental_environment;
    ValueIteration<std::vector<int>, int> value_iteration(&car_rental_environment);

    car_rental_environment.initialize();
    value_iteration.initialize();

    double time_taken = benchmark([&]() { value_iteration.policy_iteration(); });

    std::cout << "Time taken: " << time_taken << std::endl;

    auto policy = value_iteration.get_policy();

    policy.serialize_to_json("policy.json");
    plot_policy(policy);

    return 0;
}
