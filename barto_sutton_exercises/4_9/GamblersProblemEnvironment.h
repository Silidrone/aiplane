#pragma once

#include <unordered_map>
#include <vector>

#include "DeterministicPolicy.h"
#include "MDP.h"

static constexpr int MAX_STAKE = 100;
static constexpr int MIN_STAKE = 0;
static constexpr int TAILS = 0;
static constexpr int HEADS = 1;
static constexpr double P_H = 0.4;      // probability of heads ocurring
static constexpr double P_T = 1 - P_H;  // probability of tails ocurring

static std::vector<int> coin_flips = {TAILS, HEADS};

using State = int;
using Action = int;

class GamblersProblemEnvironment : public MDP<State, Action> {
   public:
    void initialize() override;
    void plot_policy(DeterministicPolicy<State, Action> &);
};
