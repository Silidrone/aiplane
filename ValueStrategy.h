#pragma once
#include <torch/torch.h>

#include <string>

#include "MDP.h"
#include "TorchModel.h"

template <typename State, typename Action>
class ValueStrategy {
   public:
    virtual ~ValueStrategy() = default;

    virtual void initialize(MDP<State, Action>* mdp) = 0;
    virtual std::tuple<Action, Return> get_best_action(const State& s) = 0;
    virtual double Q(const State& s, const Action& a) const = 0;
    virtual void update(const State& s, const Action& a, double target_q) = 0;

    virtual void save(const std::string& path) const {
        throw std::logic_error("Save not implemented for this ValueStrategy");
    }

    virtual void load(const std::string& path) {
        throw std::logic_error("Load not implemented for this ValueStrategy");
    }
};

template <typename State, typename Action>
class TabularValueStrategy : public ValueStrategy<State, Action> {
   protected:
    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>
        m_Q{};  // Action-value Q function
    MDP<State, Action>* m_mdp;
    bool m_strict{false};
    double m_step_size{0.1};  // Default step size for updates

   public:
    TabularValueStrategy(double step_size = 0.1) : m_mdp(nullptr), m_step_size(step_size) {}

    void set_strict_mode(bool strict) { m_strict = strict; }

    void set_step_size(double step_size) { m_step_size = step_size; }

    void initialize(MDP<State, Action>* mdp) override {
        m_mdp = mdp;

        for (const State& s : m_mdp->S()) {
            for (const Action& a : m_mdp->A(s)) {
                m_Q[{s, a}] = 0;
            }
        }

        for (const State& s : m_mdp->T()) {
            for (const Action& a : m_mdp->A(s)) {
                m_Q[{s, a}] = 0;
            }
        }
    }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_mdp) {
            throw std::logic_error("TabularValueStrategy not initialized with an MDP");
        }

        Return max_return = std::numeric_limits<Return>::lowest();
        Action maximizing_action;

        for (const Action& a : m_mdp->A(s)) {
            Return candidate_return = Q(s, a);
            if (candidate_return > max_return) {
                max_return = candidate_return;
                maximizing_action = a;
            }
        }

        return {maximizing_action, max_return};
    }

    double Q(const State& s, const Action& a) const override {
        auto it = m_Q.find({s, a});
        if (it == m_Q.end()) {
            if (!m_strict) return 0;
            throw std::runtime_error("Error: Invalid state-action pair provided for the Q-value function.");
        }
        return it->second;
    }

    void update(const State& s, const Action& a, double target_q) override {
        double current_q = Q(s, a);
        double updated_q = current_q + m_step_size * (target_q - current_q);
        set_q(s, a, updated_q);
    }

    void set_q(const State& s, const Action& a, Return value) { m_Q[{s, a}] = value; }

    std::unordered_map<std::pair<State, Action>, Return, StateActionPairHash<State, Action>>& get_Q() { return m_Q; }
};

template <typename State, typename Action>
class ApproximationValueStrategy : public ValueStrategy<State, Action> {
   protected:
    FunctionApproximator<State, Action>* m_approximator;
    MDP<State, Action>* m_mdp;
    double m_step_size;

   public:
    ApproximationValueStrategy(FunctionApproximator<State, Action>* approximator = nullptr, double step_size = 0.1)
        : m_approximator(approximator), m_mdp(nullptr), m_step_size(step_size) {}

    void initialize(MDP<State, Action>* mdp) override {
        if (!mdp || !m_approximator) {
            throw std::invalid_argument("Both MDP and FunctionApproximator must be non-null");
        }
        m_mdp = mdp;
    }

    void initialize(MDP<State, Action>* mdp, FunctionApproximator<State, Action>* approximator) {
        m_mdp = mdp;
        m_approximator = approximator;
    }

    void set_approximator(FunctionApproximator<State, Action>* approximator) { m_approximator = approximator; }

    void set_step_size(double step_size) { m_step_size = step_size; }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_approximator || !m_mdp) {
            throw std::logic_error("ApproximationValueStrategy not properly initialized");
        }

        Action best_action;
        double best_value = std::numeric_limits<double>::lowest();

        auto all_actions = m_mdp->all_possible_actions();
        for (Action a : all_actions) {
            if (!m_mdp->is_valid(s, a)) {
                continue;
            }

            double value = Q(s, a);

            if (value > best_value) {
                best_value = value;
                best_action = a;
            }
        }

        return {best_action, best_value};
    }

    double Q(const State& s, const Action& a) const override {
        if (!m_approximator) {
            throw std::logic_error("ApproximationValueStrategy: No approximator set");
        }
        return m_approximator->predict(s, a);
    }

    void update(const State& s, const Action& a, double target_q) override {
        if (!m_approximator) {
            throw std::logic_error("ApproximationValueStrategy: No approximator set");
        }
        double current_q = Q(s, a);
        double error = target_q - current_q;
        m_approximator->update(s, a, error, m_step_size);
    }

    FunctionApproximator<State, Action>* get_approximator() const { return m_approximator; }
};

template <typename State, typename Action>
class TorchValueStrategy : public ValueStrategy<State, Action> {
   private:
    TorchModel* q_network;
    std::function<torch::Tensor(const State&, const Action&)> feature_extractor;
    torch::optim::Adam optimizer;
    MDP<State, Action>* m_mdp;
    double m_step_size;

   public:
    TorchValueStrategy(TorchModel* network, std::function<torch::Tensor(const State&, const Action&)> fe,
                       double step_size = 0.01)
        : q_network(network),
          feature_extractor(fe),
          optimizer(network->parameters(), torch::optim::AdamOptions(step_size)),
          m_mdp(nullptr),
          m_step_size(step_size) {}

    void initialize(MDP<State, Action>* mdp) override { m_mdp = mdp; }

    std::tuple<Action, Return> get_best_action(const State& s) override {
        if (!m_mdp) {
            throw std::logic_error("TorchValueStrategy not initialized with an MDP");
        }

        Action best_action;
        double best_value = std::numeric_limits<double>::lowest();

        auto actions = m_mdp->A(s);
        for (const Action& a : actions) {
            if (m_mdp->is_valid(s, a)) {
                double q_value = Q(s, a);

                if (q_value > best_value) {
                    best_value = q_value;
                    best_action = a;
                }
            }
        }

        return {best_action, best_value};
    }

    double Q(const State& s, const Action& a) const override {
        torch::NoGradGuard no_grad;
        torch::Tensor state_action = feature_extractor(s, a);
        return q_network->forward(state_action).item<double>();
    }

    void update(const State& s, const Action& a, double target_q) override {
        torch::Tensor state_action = feature_extractor(s, a);
        torch::Tensor current_q = q_network->forward(state_action);
        torch::Tensor target = torch::tensor(target_q, torch::dtype(current_q.dtype()));

        torch::Tensor loss = torch::mse_loss(current_q, target);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    void save(const std::string& path) const override {
        // Do nothing for now since serialization is causing issues
        // We'll need to implement a proper serialization mechanism for TorchModel
    }

    void load(const std::string& path) override {
        // Do nothing for now since serialization is causing issues
        // We'll need to implement a proper serialization mechanism for TorchModel
    }
};