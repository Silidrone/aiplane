#pragma once

#include <m_utils.h>

#include <functional>
#include <tuple>
#include <unordered_map>
#include <vector>

using Reward = double;
using Return = double;
using TimeStep = int;
using Probability = double;

template <typename State, typename Action>
class MDPCore {
   public:
    using Transition = std::tuple<State, Reward, Probability>;
    using Dynamics =
        std::unordered_map<std::pair<State, Action>, std::vector<Transition>, StateActionPairHash<State, Action>>;

   protected:
    std::vector<State> m_S;                                                // State space: S
    std::unordered_map<State, std::vector<Action>, StateHash<State>> m_A;  // Action space: A
    Dynamics m_dynamics;                                                   // Dynamics p function
    TimeStep m_T;

    virtual void initialize_state_space(){};
    virtual void initialize_action_space() {}
    virtual void initialize_dynamics(){};

   public:
    virtual ~MDPCore() = default;

    MDPCore() : m_T(0) {}

    virtual void initialize() {
        initialize_state_space();
        initialize_action_space();
        initialize_dynamics();
    }

    void increment_time_step() { ++m_T; }

    [[nodiscard]] TimeStep get_time_step() const { return m_T; }

    std::vector<Transition> p(State s, Action a) { return m_dynamics[{s, a}]; }

    Dynamics dynamics() { return m_dynamics; }

    std::vector<State> S() { return m_S; }

    std::vector<Action> A(State s) { return m_A[s]; }
};
