#pragma once

#include <fstream>
#include <nlohmann/json.hpp>  // Include JSON library
#include <stdexcept>
#include <unordered_map>

#include "./m_utils.h"

template <typename State, typename Action>
class Policy {
   protected:
    std::unordered_map<State, Action, StateHash<State>> m_policy_map;

   public:
    Action operator()(const State& state) const {
        auto it = m_policy_map.find(state);
        if (it == m_policy_map.end()) {
            throw std::runtime_error("Error: Invalid state provided for the pi policy function.");
        }
        return it->second;
    }

    Action& operator[](const State& state) { return m_policy_map[state]; }

    void set(const State& state, const Action& action) { m_policy_map[state] = action; }

    bool has(const State& state) const { return m_policy_map.contains(state); }

    const std::unordered_map<State, Action, StateHash<State>>& map_container() const { return m_policy_map; }

    std::string to_json_string() const {
        nlohmann::json j;
        for (const auto& entry : m_policy_map) {
            j[std::to_string(entry.first)] = entry.second;
        }
        return j.dump(4);
    }

    void serialize_to_json(const std::string& filename) const {
        std::ofstream file(filename);
        if (file.is_open()) {
            file << to_json_string();
            file.close();
        } else {
            throw std::runtime_error("Failed to open file for writing JSON.");
        }
    }
};
