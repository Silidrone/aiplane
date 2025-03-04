#pragma once

#include <unordered_map>
#include <vector>

#include "Communicator.h"
#include "DeterministicPolicy.h"
#include "MDP.h"
#include "m_types.h"

static const std::vector<std::pair<int, int>> DIRECTION_VECTORS = {{0, 0},  {1, 0},   {1, 1},  {0, 1}, {-1, 1},
                                                                   {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
static constexpr int MIN_DISTANCE = 1;
static constexpr int MAX_DISTANCE = 2;
static const std::string TAGGAME_HOST = "127.0.0.1";
static const int TAGGAME_PORT = 12345;

// (taggedVelocity, myVelocity, distance)
using State = std::tuple<std::pair<int, int>, std::pair<int, int>, int>;
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
    std::string serialize_action(Action);
    State deserialize_state(const std::string &);
    Reward calculate_reward(const State &, const State &);
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;
    std::vector<Action> all_actions();
    void plot_policy(DeterministicPolicy<State, Action> &);
};
