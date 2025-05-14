#pragma once
#include <MDPSolver.h>
#include <ValueStrategy.h>

#include <unordered_map>

#include "m_types.h"

template <typename State, typename Action>
class Policy {
   protected:
    MDP<State, Action>* m_mdp;
    ValueStrategy<State, Action>* m_value_strategy;

   public:
    Policy() : m_mdp(nullptr), m_value_strategy(nullptr) {}

    Policy(ValueStrategy<State, Action>* value_strategy) : m_mdp(nullptr), m_value_strategy(value_strategy) {}

    virtual ~Policy() = default;

    void initialize(MDP<State, Action>* mdp, ValueStrategy<State, Action>* strategy) {
        if (!strategy) {
            throw std::invalid_argument("MDP cannot be null!");
        }

        if (!strategy) {
            throw std::invalid_argument("Value strategy cannot be null!");
        }

        m_mdp = mdp;
        m_value_strategy = strategy;
    }

    virtual Action sample(const State& s) { return std::get<0>(greedy_action(s)); }

    virtual std::tuple<Action, Return> greedy_action(const State& s) { return m_value_strategy->get_best_action(s); }

    std::unordered_map<State, Action, StateHash<State>> optimal() const {
        if (!m_mdp || !m_value_strategy) {
            throw std::logic_error("Policy not properly initialized with MDP and ValueStrategy");
        }

        std::unordered_map<State, Action, StateHash<State>> optimal_policy;

        for (const State& s : m_mdp->S()) {
            if (!m_mdp->is_terminal(s)) {
                auto [best_action, _] = const_cast<Policy*>(this)->greedy_action(s);
                optimal_policy[s] = best_action;
            }
        }

        return optimal_policy;
    }

    ValueStrategy<State, Action>* value_strategy() { return m_value_strategy; }
};

template <typename State, typename Action>
class DeterministicPolicy : public Policy<State, Action> {
   private:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

   public:
    DeterministicPolicy() : Policy<State, Action>() {}

    DeterministicPolicy(ValueStrategy<State, Action>* value_strategy) : Policy<State, Action>(value_strategy) {}

    void set(const State& state, const Action& action) { m_policy_map[state] = action; }

    Action sample(const State& s) override {
        auto it = m_policy_map.find(s);
        if (it != m_policy_map.end()) {
            return it->second;
        }
        return std::get<0>(this->greedy_action(s));
    }
};

template <typename State, typename Action>
class EpsilonGreedyPolicy : public Policy<State, Action> {
   private:
    double m_epsilon;
    std::mt19937 m_generator;
    double m_min_epsilon;  // Minimum epsilon value for decay
    double m_epsilon_decay;

   public:
    EpsilonGreedyPolicy(ValueStrategy<State, Action>* value_strategy, double epsilon, double min_epsilon = 0.01,
                        double epsilon_decay = 1.0)
        : Policy<State, Action>(value_strategy),
          m_epsilon(epsilon),
          m_generator(std::random_device{}()),
          m_min_epsilon(min_epsilon),
          m_epsilon_decay(epsilon_decay) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
        if (min_epsilon < 0.0 || min_epsilon > epsilon) {
            throw std::invalid_argument("Min epsilon must be between 0 and epsilon");
        }
        if (epsilon_decay <= 0.0 || epsilon_decay > 1.0) {
            throw std::invalid_argument("Epsilon decay must be between 0 and 1");
        }
    }

    Action sample(const State& s) override {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(m_generator) < m_epsilon) {
            auto actions = this->m_mdp->A(s);
            if (actions.empty()) {
                throw std::runtime_error("No available actions for the given state");
            }
            std::uniform_int_distribution<int> action_dist(0, actions.size() - 1);
            return actions[action_dist(m_generator)];
        } else {
            return std::get<0>(this->greedy_action(s));
        }
    }

    double get_epsilon() const { return m_epsilon; }

    void set_epsilon(double epsilon) {
        if (epsilon < 0.0 || epsilon > 1.0) {
            throw std::invalid_argument("Epsilon must be between 0 and 1");
        }
        m_epsilon = epsilon;
    }

    double get_min_epsilon() const { return m_min_epsilon; }

    void set_min_epsilon(double min_epsilon) {
        if (min_epsilon < 0.0 || min_epsilon > 1.0) {
            throw std::invalid_argument("Min epsilon must be between 0 and 1");
        }
        m_min_epsilon = min_epsilon;
    }

    double get_epsilon_decay() const { return m_epsilon_decay; }

    void set_epsilon_decay(double epsilon_decay) {
        if (epsilon_decay <= 0.0 || epsilon_decay > 1.0) {
            throw std::invalid_argument("Epsilon decay must be between 0 and 1");
        }
        m_epsilon_decay = epsilon_decay;
    }

    // Decay epsilon (call this after each episode)
    void decay_epsilon() { m_epsilon = std::max(m_min_epsilon, m_epsilon * m_epsilon_decay); }
};