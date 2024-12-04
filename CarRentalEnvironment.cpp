#include "CarRentalEnvironment.h"
#include "m_utils.h"

void CarRentalEnvironment::generate_requests_returns_combinations(const State& s)
{
    // Generate all (R1, r1) pairs for location 1
    std::vector<std::pair<int, int>> loc1_rr;
    for (int i = 0; i <= 4; i++)
    {
        for (int j = 0; j <= 4; j++)
        {
            loc1_rr.emplace_back(i, j); // (R1, r1) pairs
        }
    }

    // Generate all (R2, r2) pairs for location 2
    std::vector<std::pair<int, int>> loc2_rr;
    for (int i = 0; i <= 5; i++)
    {
        for (int j = 0; j <= 3; j++)
        {
            loc2_rr.emplace_back(i, j); // (R2, r2) pairs
        }
    }

    // Now generate the Cartesian product of loc1_rr and loc2_rr
    for (const auto &rr1 : loc1_rr)
    {
        for (const auto &rr2 : loc2_rr)
        {
            m_rrs[s].push_back({rr1, rr2}); // Combine each (R1, r1) with each (R2, r2) if probability > 5%
        }
    }
}

void CarRentalEnvironment::generate_action_space()
{
    for (int i = -5; i <= 5; i++)
    {
        m_A.emplace_back(i);
    }
}

void CarRentalEnvironment::generate_dynamics_p()
{
    for (const State& s : m_S)
    {
        for (Action a : m_A)
        {
            auto state_rrs = m_rrs[s];

            for (auto & state_rr : state_rrs)
            {
                State s_prime = s;
                Reward reward = 0;

                // move the cars
                reward -= abs(a * COST_OF_MOVING_A_CAR);
                s_prime[0] -= a;
                s_prime[1] += a;

                double probability = 1;
                for (int loc = 0; loc < s.size(); loc++)
                {
                    auto [requested, returned] = state_rr[loc];
                    probability *= poisson_probability(requested, requests_lambda[loc]) *
                                   poisson_probability(returned, returns_lambda[loc]);

                    // rent out the cars
                    const int fulfilled_requests = std::min(requested, s_prime[loc]);
                    reward += fulfilled_requests * REQUEST_FULFILLMENT_REWARD;
                    s_prime[loc] -= fulfilled_requests;

                    // return the cars
                    s_prime[loc] += returned;
                }

                m_dynamics[{s, a}].emplace_back(s_prime, reward, probability);
            }
        }
    }
}

void CarRentalEnvironment::initialize() {
    for (int i = 0; i <= MAX_CARS_COUNT_PER_LOCATION; i++)
    {
        for (int j = 0; j <= MAX_CARS_COUNT_PER_LOCATION; j++)
        {
            State s = {i, j};
            m_S.push_back(s);
            generate_requests_returns_combinations(s);
        }
    }

    generate_action_space();
    generate_dynamics_p();
}
