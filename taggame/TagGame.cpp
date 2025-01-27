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

    for (int rotation = -MAX_ANGLE; rotation <= MAX_ANGLE; rotation += ANGLE_SENSITIVITY) {
        for (int speed = MIN_SPEED; speed <= MAX_SPEED; speed += SPEED_SENSITIVITY) {
            m_all_actions.emplace_back(std::make_tuple(rotation, speed));
        }
    }
}

bool TagGame::is_terminal(const State& s) { return std::get<3>(s); }

State TagGame::deserialize_state(const std::string& str_state) {
    try {
        nlohmann::json gameState = nlohmann::json::parse(str_state);

        Eigen::Vector2d myVelocity(gameState["mv"][0], gameState["mv"][1]);
        Eigen::Vector2d taggedVelocity(gameState["tv"][0], gameState["tv"][1]);
        int distance = gameState["d"];
        bool tag_changed = gameState["tc"];

        std::cout << "Parsed Data:" << std::endl;
        std::cout << "My Velocity: [" << myVelocity.transpose() << "]" << std::endl;
        std::cout << "Tagged Velocity: [" << taggedVelocity.transpose() << "]" << std::endl;
        std::cout << "Distance: [" << distance << "]" << std::endl;
        std::cout << "Tag changed: " << (tag_changed ? "true" : "false") << std::endl;

        return {myVelocity, taggedVelocity, distance, tag_changed};
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
    }
}

std::string TagGame::serialize_action(Action a) {
    nlohmann::json serialized_action;

    int angle_diff = std::get<0>(a);
    int speed = std::get<1>(a);

    serialized_action["a"] = angle_diff;
    serialized_action["s"] = speed;

    return serialized_action.dump();
}

Reward TagGame::calculate_reward(const State& old_s, const State& new_s) {
    auto [old_tagged_velocity, old_my_velocity, old_distance, _] = old_s;
    auto [new_tagged_velocity, new_my_velocity, new_distance, tag_changed] = new_s;

    if (tag_changed) {  // terminal: the tagged player changed
        if (old_distance == 0 && new_distance != 0) {
            return CATCH_REWARD;  // Caught someone
        } else if (old_distance != 0 && new_distance == 0) {
            return CAUGHT_REWARD;  // Got caught
        } else if (old_distance == 0 && new_distance == 0) {
            return STAY_TAGGED_REWARD;  // Staying tagged
        } else if (old_distance != 0 && new_distance != 0) {
            return STAY_UNTAGGED_REWARD;  // Staying untagged
        }
    } else {
        if (old_distance > DISTANCE_THRESHOLD && new_distance <= DISTANCE_THRESHOLD) {
            return GET_CLOSER_REWARD;
        } else if (old_distance <= DISTANCE_THRESHOLD && new_distance > DISTANCE_THRESHOLD) {
            return GET_FURTHER_REWARD;
        }
    }

    return 0;
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