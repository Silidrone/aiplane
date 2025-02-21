#include "TagGame.h"

#include <math.h>
#include <matplot/matplot.h>

#include <string>

#include "m_types.h"
#include "m_utils.h"
#include "taggame/TagGame.h"

void TagGame::initialize() {
    if (!m_communicator.connectToServer(TAGGAME_HOST, TAGGAME_PORT)) {
        throw std::runtime_error(
            "Failed to initialize: Failed to connect to the TagGame! Please run the TagGame first and then the RL "
            "control.");
    }

    for (const auto& mv : DIRECTION_VECTORS) {
        for (const auto& tv : DIRECTION_VECTORS) {
            for (int distance = MIN_DISTANCE; distance <= MAX_DISTANCE; ++distance) {
                for (int i = 0; i <= 1; i++) {  // isTouchingWall: true, false
                    State s = std::make_tuple(mv, tv, distance, i);
                    m_S.push_back(s);
                    m_A[s] = std::vector<std::pair<int, int>>(DIRECTION_VECTORS.begin() + 1, DIRECTION_VECTORS.end());
                }
            }
        }
    }
}

std::string TagGame::serialize_action(Action a) {
    nlohmann::json serialized_action;

    int x = std::get<0>(a);
    int y = std::get<1>(a);

    serialized_action["x"] = x;
    serialized_action["y"] = y;

    return serialized_action.dump();
}

State TagGame::deserialize_state(const std::string& str_state) {
    try {
        nlohmann::json gameState = nlohmann::json::parse(str_state);

        std::pair<int, int> myVelocity(gameState["mv"][0], gameState["mv"][1]);
        std::pair<int, int> taggedVelocity(gameState["tv"][0], gameState["tv"][1]);
        int distance = gameState["d"];
        bool isTouchingWall = std::stoi(static_cast<std::string>(gameState["tw"])) != 0;

        std::cout << "Received: ([" << myVelocity.first << ", " << myVelocity.second << "], [" << taggedVelocity.first
                  << ", " << taggedVelocity.second << "], " << distance << ", " << isTouchingWall << ")" << std::endl;

        return {myVelocity, taggedVelocity, distance, isTouchingWall};
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
}

bool TagGame::is_terminal(const State& s) { return std::get<2>(s) == 0; }

Reward TagGame::calculate_reward(const State& old_s, const State& new_s) {
    auto [old_my_velocity, old_tagged_velocity, old_distance, old_touching_wall] = old_s;
    auto [new_my_velocity, new_tagged_velocity, new_distance, new_touching_wall] = new_s;

    if (new_distance == MIN_DISTANCE || new_touching_wall) {
        return -1000000;
    }

    return 1;
}

State TagGame::reset() {
    m_communicator.sendAction(m_communicator.RESET);
    return deserialize_state(m_communicator.receiveState());
}

std::pair<State, Reward> TagGame::step(const State& old_s, const Action& action) {
    m_communicator.sendAction(serialize_action(action));
    State new_s = deserialize_state(m_communicator.receiveState());

    // std::cout << "action: " << action.first << action.second << std::endl;

    return {new_s, calculate_reward(old_s, new_s)};
}

std::vector<Action> TagGame::all_actions() { return m_all_actions; }

void TagGame::plot_policy(DeterministicPolicy<State, Action>& pi) {}