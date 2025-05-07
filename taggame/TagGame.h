#pragma once

#include <unordered_map>
#include <vector>

#include "Communicator.h"
#include "MDP.h"
#include "Policy.h"
#include "m_types.h"

static constexpr double MAX_VELOCITY = 10;
static constexpr double MAX_X = 1000;
static constexpr double MAX_Y = 1000;
static const double MAX_DISTANCE = std::sqrt(MAX_X * MAX_X + MAX_Y * MAX_Y);

static const std::string TAGGAME_HOST = "127.0.0.1";
static const int TAGGAME_PORT = 12345;

// (myPosition, myVelocity, tagPosition, tagVelocity, isTagged)
using State = std::tuple<std::pair<int, int>, std::pair<int, int>, std::pair<int, int>, std::pair<int, int>, bool>;
// the x and y components of the velocity vector
using Action = std::pair<int, int>;

class TagGame : public MDP<State, Action> {
   protected:
    Communicator &m_communicator;
    std::vector<Action> m_all_actions;

   public:
    TagGame() : MDP(), m_communicator(Communicator::getInstance()) {}
    virtual ~TagGame() { Communicator::getInstance().disconnect(); };
    void initialize() override;
    bool is_terminal(const State &s) override;
    bool is_valid(const State &s, const Action &a) const override { return true; };
    std::string serialize_action(Action);
    State deserialize_state(const std::string &);
    Reward calculate_reward(const State &, const State &);
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;
    std::vector<Action> all_possible_actions() const override;
    void plot_policy(DeterministicPolicy<State, Action> &);
};
