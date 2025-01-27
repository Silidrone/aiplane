#pragma once

#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <vector>

#include "Communicator.h"
#include "DeterministicPolicy.h"
#include "MDP.h"
#include "m_types.h"

static constexpr double DISCOUNT_RATE = 1;  // no discounting

static constexpr Reward GET_CLOSER_REWARD = -1;
static constexpr Reward GET_FURTHER_REWARD = 1;

static constexpr Reward STAY_TAGGED_REWARD = -5;
static constexpr Reward STAY_UNTAGGED_REWARD = 5;

static constexpr Reward CAUGHT_REWARD = -10;
static constexpr Reward CATCH_REWARD = 10;

static constexpr int MAX_SPEED = 10;
static constexpr int MIN_SPEED = 0;
static constexpr int MAX_ANGLE = 45;
static constexpr int ANGLE_SENSITIVITY = 15;
static constexpr int SPEED_SENSITIVITY = 1;
static constexpr int WINDOW_WIDTH = 800;
static constexpr int WINDOW_HEIGHT = 600;
static constexpr int DISTANCE_THRESHOLD = 100;
static const std::string TAGGAME_HOST = "127.0.0.1";
static const int TAGGAME_PORT = 12345;

// (taggedVelocity, myVelocity, distance, tagChanged) all vectors can only take whole numbers, and magnitude can be at
// maximum between MIN_SPEED and MAX_SPEED
using State = std::tuple<Eigen::Vector2d, Eigen::Vector2d, int, bool>;
// rotation: between -45 and 45 (can move by 15, so 6 values) and speed: between 0 and 10
using Action = std::tuple<int, int>;

class TagGame : public MDP<State, Action> {
   protected:
    Communicator &m_communicator;
    std::vector<Action> m_all_actions;

   public:
    TagGame() : m_communicator(Communicator::getInstance()) {}
    virtual ~TagGame() { Communicator::getInstance().disconnect(); };
    void initialize() override;
    bool is_terminal(const State &s) override;
    State deserialize_state(const std::string &);
    std::string serialize_action(Action);
    Reward calculate_reward(const State &, const State &);
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;
    std::vector<Action> all_actions();
    void plot_policy(DeterministicPolicy<State, Action> &);
};
