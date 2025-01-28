#pragma once

#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <vector>

#include "Communicator.h"
#include "DeterministicPolicy.h"
#include "MDP.h"
#include "m_types.h"

static constexpr double DISCOUNT_RATE = 1;  // no discounting

static constexpr Reward STAY_UNTAGGED_REWARD = 1;
static constexpr Reward GET_CAUGHT_REWARD = -600;

static constexpr int MAX_VELOCITY = 3;
static const std::string TAGGAME_HOST = "127.0.0.1";
static const int TAGGAME_PORT = 12345;

// (taggedVelocity, myVelocity, distance, tagChanged)
using State = std::tuple<Eigen::Vector2d, Eigen::Vector2d, int>;
// the x and y components of the velocity vector
using Action = std::tuple<int, int>;

class TagGame : public MDP<State, Action> {
   protected:
    Communicator &m_communicator;
    std::vector<Action> m_all_actions;

   public:
    TagGame() : MDP(true), m_communicator(Communicator::getInstance()) {}
    virtual ~TagGame() { Communicator::getInstance().disconnect(); };
    void initialize() override;
    State deserialize_state(const std::string &);
    std::string serialize_action(Action);
    Reward calculate_reward(const State &, const State &);
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;
    std::vector<Action> all_actions();
    void plot_policy(DeterministicPolicy<State, Action> &);
};
