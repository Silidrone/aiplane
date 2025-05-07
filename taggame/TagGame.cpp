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

    // Initialize all possible actions once
    m_all_actions.clear();
    for (int ax = -MAX_VELOCITY; ax <= MAX_VELOCITY; ++ax) {
        for (int ay = -MAX_VELOCITY; ay <= MAX_VELOCITY; ++ay) {
            if (ax != 0 || ay != 0) {
                m_all_actions.push_back({ax, ay});
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

        std::pair<int, int> myPosition(gameState["mp"][0], gameState["mp"][1]);
        std::pair<int, int> myVelocity(gameState["mv"][0], gameState["mv"][1]);
        std::pair<int, int> tagPosition(gameState["tp"][0], gameState["tp"][1]);
        std::pair<int, int> tagVelocity(gameState["tv"][0], gameState["tv"][1]);
        bool isTagged = gameState["t"];

        // std::cout << "Received: mp=[" << myPosition.first << ", " << myPosition.second << "], mv=[" <<
        // myVelocity.first
        //           << ", " << myVelocity.second << "], tp=[" << tagPosition.first << ", " << tagPosition.second
        //           << "], tv=[" << tagVelocity.first << ", " << tagVelocity.second << "], tagged=" << isTagged
        //           << std::endl;

        return {myPosition, myVelocity, tagPosition, tagVelocity, isTagged};
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        throw;
    }
}

bool TagGame::is_terminal(const State& s) { return std::get<4>(s); }

Reward TagGame::calculate_reward(const State& old_s, const State& new_s) {
    auto [old_my_pos, old_my_vel, old_tag_pos, old_tag_vel, old_is_tagged] = old_s;
    auto [new_my_pos, new_my_vel, new_tag_pos, new_tag_vel, new_is_tagged] = new_s;

    if (new_is_tagged) {
        return -10.0;
    }

    double old_distance = std::sqrt(std::pow(old_my_pos.first - old_tag_pos.first, 2) +
                                    std::pow(old_my_pos.second - old_tag_pos.second, 2)) /
                          MAX_DISTANCE;

    double new_distance = std::sqrt(std::pow(new_my_pos.first - new_tag_pos.first, 2) +
                                    std::pow(new_my_pos.second - new_tag_pos.second, 2)) /
                          MAX_DISTANCE;

    double distance_reward = 0.5 * (new_distance - old_distance);

    std::vector<std::pair<int, int>> corners = {
        {0, 0},                                             // Bottom left
        {0, static_cast<int>(MAX_Y)},                       // Top left
        {static_cast<int>(MAX_X), 0},                       // Bottom right
        {static_cast<int>(MAX_X), static_cast<int>(MAX_Y)}  // Top right
    };

    double min_old_corner_dist = MAX_DISTANCE;
    double min_new_corner_dist = MAX_DISTANCE;

    for (const auto& corner : corners) {
        double old_corner_dist =
            std::sqrt(std::pow(old_my_pos.first - corner.first, 2) + std::pow(old_my_pos.second - corner.second, 2));
        min_old_corner_dist = std::min(min_old_corner_dist, old_corner_dist);

        double new_corner_dist =
            std::sqrt(std::pow(new_my_pos.first - corner.first, 2) + std::pow(new_my_pos.second - corner.second, 2));
        min_new_corner_dist = std::min(min_new_corner_dist, new_corner_dist);
    }

    min_old_corner_dist /= MAX_DISTANCE;
    min_new_corner_dist /= MAX_DISTANCE;

    double corner_reward = 0.3 * (min_new_corner_dist - min_old_corner_dist);

    double survival_reward = 0.01;

    return distance_reward + corner_reward + survival_reward;
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

std::vector<Action> TagGame::all_possible_actions() const { return m_all_actions; }

void TagGame::plot_policy(DeterministicPolicy<State, Action>& pi) {}