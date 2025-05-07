#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

#include "TagGame.h"

namespace {
// Vector2D class for vector operations
class Vector2D {
   private:
    double x, y;

   public:
    Vector2D() : x(0), y(0) {}
    Vector2D(double x, double y) : x(x), y(y) {}
    Vector2D(const std::pair<int, int>& from, const std::pair<int, int>& to)
        : x(to.first - from.first), y(to.second - from.second) {}

    double length() const { return std::sqrt(x * x + y * y); }

    Vector2D normalize() const {
        double len = length();
        if (len < 0.0001) return Vector2D(0, 0);
        return Vector2D(x / len, y / len);
    }

    Vector2D times(double scalar) const { return Vector2D(x * scalar, y * scalar); }

    Vector2D plus(const Vector2D& other) const { return Vector2D(x + other.x, y + other.y); }

    std::pair<int, int> toIntPair() const {
        return std::make_pair(static_cast<int>(std::round(x)), static_cast<int>(std::round(y)));
    }

    double getX() const { return x; }
    double getY() const { return y; }
};

// Calculate distance between two points
double distance(const std::pair<int, int>& p1, const std::pair<int, int>& p2) {
    double dx = p1.first - p2.first;
    double dy = p1.second - p2.second;
    return std::sqrt(dx * dx + dy * dy);
}

// Get all corners of the game arena
std::vector<std::pair<int, int>> getCorners() {
    return {
        {0, 0},                                             // Bottom left
        {0, static_cast<int>(MAX_Y)},                       // Top left
        {static_cast<int>(MAX_X), 0},                       // Bottom right
        {static_cast<int>(MAX_X), static_cast<int>(MAX_Y)}  // Top right
    };
}

// Order corners by distance from the predator (ascending)
std::vector<std::pair<std::pair<int, int>, double>> orderCornersByDistance(const std::pair<int, int>& predatorPos) {
    std::vector<std::pair<std::pair<int, int>, double>> orderedCorners;
    std::vector<std::pair<int, int>> corners = getCorners();

    for (const auto& corner : corners) {
        double dist = distance(predatorPos, corner);
        orderedCorners.push_back({corner, dist});
    }

    // Sort by distance (ascending)
    std::sort(orderedCorners.begin(), orderedCorners.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    return orderedCorners;
}
}  // namespace

class DeterministicSolution {
   private:
    Communicator& m_communicator;
    const double safeDistanceThreshold;
    const double maxVelocity;

   public:
    DeterministicSolution()
        : m_communicator(Communicator::getInstance()),
          safeDistanceThreshold(MAX_DISTANCE * 0.3),  // 30% of max distance is considered "safe"
          maxVelocity(MAX_VELOCITY) {}

    ~DeterministicSolution() { m_communicator.disconnect(); }

    void run() {
        if (!m_communicator.connectToServer(TAGGAME_HOST, TAGGAME_PORT)) {
            throw std::runtime_error("Failed to connect to the TagGame! Please run the TagGame first.");
        }

        std::cout << "Starting deterministic solution..." << std::endl;

        // Reset game state
        m_communicator.sendAction(m_communicator.RESET);
        std::string state_str = m_communicator.receiveState();

        while (true) {
            try {
                // Parse game state
                nlohmann::json gameState = nlohmann::json::parse(state_str);

                std::pair<int, int> myPosition(gameState["mp"][0], gameState["mp"][1]);
                std::pair<int, int> myVelocity(gameState["mv"][0], gameState["mv"][1]);
                std::pair<int, int> tagPosition(gameState["tp"][0], gameState["tp"][1]);
                std::pair<int, int> tagVelocity(gameState["tv"][0], gameState["tv"][1]);
                bool isTagged = gameState["t"];

                // If tagged, game is over
                if (isTagged) {
                    std::cout << "Tagged! Game over." << std::endl;
                    // Reset for next game
                    m_communicator.sendAction(m_communicator.RESET);
                    state_str = m_communicator.receiveState();
                    continue;
                }

                // Compute deterministic action based on the Java algorithm
                auto action = computeAction(myPosition, tagPosition);

                // Send action to the game
                nlohmann::json actionJson;
                actionJson["x"] = action.first;
                actionJson["y"] = action.second;

                m_communicator.sendAction(actionJson.dump());

                // Receive new state
                state_str = m_communicator.receiveState();

            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << std::endl;
                break;
            }
        }
    }

   private:
    std::pair<int, int> computeAction(const std::pair<int, int>& myPosition,
                                      const std::pair<int, int>& opponentPosition) {
        // Order corners by distance from the predator
        auto orderedCorners = orderCornersByDistance(opponentPosition);

        // Calculate distance to opponent
        double distanceToOpponent = distance(myPosition, opponentPosition);

        // Calculate weights for fleeing vs. seeking based on opponent distance
        double fleeWeight = std::max(0.0, (safeDistanceThreshold - distanceToOpponent) / safeDistanceThreshold);
        double seekWeight = 1.0 - fleeWeight;

        // Get furthest corner from opponent (last in ordered list)
        std::pair<int, int> furthestCornerFromOpponent = orderedCorners.back().first;

        // Calculate vectors
        Vector2D desiredDirection(myPosition, furthestCornerFromOpponent);
        Vector2D fromMeToOpponent(myPosition, opponentPosition);

        // Normalize vectors
        desiredDirection = desiredDirection.normalize();
        fromMeToOpponent = fromMeToOpponent.normalize();

        // Calculate desired velocity
        Vector2D desiredVelocity = desiredDirection.times(seekWeight).plus(fromMeToOpponent.times(-fleeWeight));

        // Normalize and scale to max velocity
        desiredVelocity = desiredVelocity.normalize().times(maxVelocity);

        return desiredVelocity.toIntPair();
    }
};

inline int taggame_main() {
    try {
        DeterministicSolution solution;
        solution.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in deterministic solution: " << e.what() << std::endl;
        return 1;
    }
}