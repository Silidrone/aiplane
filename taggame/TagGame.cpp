#include "TagGame.h"

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

    for (int vx = -MAX_VELOCITY; vx <= MAX_VELOCITY; ++vx) {
        for (int vy = -MAX_VELOCITY; vy <= MAX_VELOCITY; ++vy) {
            if (vx * vx + vy * vy <= MAX_VELOCITY * MAX_VELOCITY) {
                m_all_actions.emplace_back(std::make_tuple(vx, vy));
            }
        }
    }
}

State TagGame::deserialize_state(const std::string& str_state) {
    try {
        nlohmann::json gameState = nlohmann::json::parse(str_state);

        std::pair<int, int> myVelocity(gameState["mv"][0], gameState["mv"][1]);
        std::pair<int, int> taggedVelocity(gameState["tv"][0], gameState["tv"][1]);
        int distance = gameState["d"];

        // std::cout << "Received: ([" << myVelocity.first << ", " << myVelocity.second << "], [" <<
        // taggedVelocity.first
        //           << ", " << taggedVelocity.second << "], " << distance << std::endl;

        return {myVelocity, taggedVelocity, distance};
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
}

bool TagGame::is_terminal(const State& s) { return std::get<2>(s) == 0; }

std::string TagGame::serialize_action(Action a) {
    nlohmann::json serialized_action;

    int x = std::get<0>(a);
    int y = std::get<1>(a);

    serialized_action["x"] = x;
    serialized_action["y"] = y;

    return serialized_action.dump();
}

Reward TagGame::calculate_reward(const State& old_s, const State& new_s) {
    auto [old_tagged_velocity, old_my_velocity, old_distance] = old_s;
    auto [new_tagged_velocity, new_my_velocity, new_distance] = new_s;

    // if (old_distance != 0 && new_distance == 0) {
    //     return GET_CAUGHT_REWARD;
    // }
    // else if (new_distance != 0) {
    //     Reward r = 0;
    //     if (new_distance > DISTANCE_THRESHOLD) {
    //         r += STAY_FURTHER_REWARD;
    //     } else {
    //         r += STAY_CLOSER_REWARD;
    //     }

    //     return r;
    // }

    Reward r = (new_distance - old_distance) * 1;

    if (new_distance < DISTANCE_THRESHOLD) {
        r += STAY_CLOSER_REWARD;
    }

    return r;
}

State TagGame::reset() {
    m_communicator.sendAction(m_communicator.RESET);
    return deserialize_state(m_communicator.receiveState());
}

std::pair<State, Reward> TagGame::step(const State& old_s, const Action& action) {
    m_communicator.sendAction(serialize_action(action));
    State new_s = deserialize_state(m_communicator.receiveState());

    return {new_s, calculate_reward(old_s, new_s)};
}

std::vector<Action> TagGame::all_actions() { return m_all_actions; }

void TagGame::plot_policy(DeterministicPolicy<State, Action>& pi) {}