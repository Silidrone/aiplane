import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mdp import MDP, Action, Reward, State
from policy import DeterministicPolicy
from environments.taggame.communicator import Communicator
from environments.taggame.taggame_constants import MAX_DISTANCE, MAX_VELOCITY, MAX_X, MAX_Y, TAGGAME_HOST, TAGGAME_PORT

Position = Tuple[int, int]
Velocity = Tuple[int, int]
TagGameState = Tuple[Position, Velocity, Position, Velocity, bool]
TagGameAction = Tuple[int, int]


class TagGame(MDP[TagGameState, TagGameAction]):
    def __init__(self):
        super().__init__()
        self._communicator = Communicator.get_instance()
        self._all_actions: List[TagGameAction] = []
    
    def initialize(self) -> None:
        if not self._communicator.connect_to_server(TAGGAME_HOST, TAGGAME_PORT):
            raise RuntimeError(
                "Failed to initialize: Failed to connect to the TagGame! "
                "Please run the TagGame first and then the RL control."
            )
        
        self._all_actions = []
        for ax in range(-MAX_VELOCITY, int(MAX_VELOCITY) + 1):
            for ay in range(-MAX_VELOCITY, int(MAX_VELOCITY) + 1):
                if ax != 0 or ay != 0:
                    self._all_actions.append((ax, ay))
    
    def serialize_action(self, action: TagGameAction) -> str:
        x, y = action
        serialized_action = {
            "x": x,
            "y": y
        }
        return json.dumps(serialized_action)
    
    def deserialize_state(self, state_str: str) -> TagGameState:
        try:
            game_state = json.loads(state_str)
            
            my_position = (game_state["mp"][0], game_state["mp"][1])
            my_velocity = (game_state["mv"][0], game_state["mv"][1])
            tag_position = (game_state["tp"][0], game_state["tp"][1])
            tag_velocity = (game_state["tv"][0], game_state["tv"][1])
            is_tagged = game_state["t"]
            
            return (my_position, my_velocity, tag_position, tag_velocity, is_tagged)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Error parsing state JSON: {e}")
    
    def is_terminal(self, state: TagGameState) -> bool:
        _, _, _, _, is_tagged = state
        return is_tagged
    
    def is_valid(self, state: TagGameState, action: TagGameAction) -> bool:
        return True
    
    def calculate_reward(self, old_state: TagGameState, new_state: TagGameState) -> float:
        _, _, _, _, old_is_tagged = old_state
        _, _, _, _, new_is_tagged = new_state
        
        if new_is_tagged:
            return -1.0
        
        return 0.01
    
    def reset(self) -> TagGameState:
        self._communicator.send_action(self._communicator.RESET)
        return self.deserialize_state(self._communicator.receive_state())
    
    def step(self, state: TagGameState, action: TagGameAction) -> Tuple[TagGameState, Reward]:
        self._communicator.send_action(self.serialize_action(action))
        new_state = self.deserialize_state(self._communicator.receive_state())
        
        return new_state, self.calculate_reward(state, new_state)
    
    def all_possible_actions(self) -> List[TagGameAction]:
        return self._all_actions
    
    def plot_policy(self, policy: DeterministicPolicy[TagGameState, TagGameAction]) -> None:
        print("Policy visualization is not implemented in the Python version")