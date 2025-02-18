#pragma once

#include <unordered_map>
#include <vector>

#include "DeterministicPolicy.h"
#include "MDP.h"
#include "m_types.h"

static constexpr double DISCOUNT_RATE = 1;  // no discounting
static constexpr Reward WIN_REWARD = 1;
static constexpr Reward LOSS_REWARD = -1;
static constexpr Reward NO_REWARD = 0;
static constexpr int ACE = 1;
static constexpr int FACE_CARD = 10;
static constexpr int USABLE_ACE_VALUE_DIFF = 10;
static constexpr int MAX_SUM = 21;
static constexpr int MIN_PLAYER_SUM = 12;
static constexpr int DEALER_POLICY_THRESHOLD = 17;

// (player's sum 12-21, dealer's facing card ace-10, usable ace present or not)
using State = std::tuple<int, int, bool>;
using Action = bool;

static const State dummy_terminal_state = {0, 0, false};

class Blackjack : public MDP<State, Action> {
   protected:
    int m_dealer_sum;
    bool m_dealer_has_usable_ace;
    int draw_card();
    std::tuple<int, bool, int> draw_card_with_checks(int, bool);

   public:
    void initialize() override;
    bool is_terminal(const State &s) override;
    State reset() override;
    std::pair<State, Reward> step(const State &, const Action &) override;

    void plot_policy(const std::unordered_map<State, Action, StateHash<State>> &, bool);
};
