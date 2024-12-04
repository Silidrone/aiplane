#ifndef CARRENTALENVIRONMENT_H
#define CARRENTALENVIRONMENT_H

#include "MDPCore.h"
#include <unordered_map>
#include <vector>

// Define constants used in the environment
static constexpr double DISCOUNT_RATE = 0.9f;
static constexpr double requests_lambda[] = {3, 4};
static constexpr double returns_lambda[] = {3, 2};
static constexpr int MAX_CARS_COUNT_PER_LOCATION = 20;
static constexpr int REQUEST_FULFILLMENT_REWARD = 10;
static constexpr int COST_OF_MOVING_A_CAR = 2;
static constexpr int NUMBER_OF_LOCATIONS = 2;
static constexpr double POLICY_THRESHOLD_EPSILON = 0.01;

using State = std::vector<int>;
using Action = int;

class CarRentalEnvironment : public MDPCore<State, Action> {
protected:
    // Requests - responses for each state
    std::unordered_map<State, std::vector<std::vector<std::pair<int, int>>>, StateHash<State>> m_rrs;

    void generate_requests_returns_combinations(const State& s);
    void generate_action_space();
    void generate_dynamics_p();
public:
    void initialize() override;
};

#endif // CARRENTALENVIRONMENT_H
