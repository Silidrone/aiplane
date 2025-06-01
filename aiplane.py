import time
from typing import Tuple, List

from mdp import MDP, Reward
from aiplane_models import AiplaneState, AiplaneAction

class AiplaneEnv(MDP[AiplaneState, AiplaneAction]):
    def __init__(self, plugin_interface, is_continuous: bool = True):
        super().__init__(is_continuous=is_continuous)
        self.plugin = plugin_interface
        self.action_space = self._generate_action_space()
        
    def _generate_action_space(self) -> List[AiplaneAction]:
        # Discrete action space for SARSA
        elevators = [-0.5, -0.2, 0.0, 0.2, 0.5]
        throttles = [0.0, 0.3, 0.5, 0.7, 1.0]
        ailerons = [-0.3, -0.1, 0.0, 0.1, 0.3]
        flaps = [0.0, 0.5, 1.0]
        
        actions = []
        for elev in elevators:
            for throttle in throttles:
                for aileron in ailerons:
                    for flap in flaps:
                        actions.append((elev, throttle, aileron, flap))
        return actions

    def initialize(self) -> None:
        pass

    def reset(self) -> AiplaneState:
        self.plugin.reset_to_approach()
        return self.plugin.read_state()

    def step(self, state: AiplaneState, action: AiplaneAction) -> Tuple[AiplaneState, Reward]:
        elevator, throttle, aileron, flaps = action
        self.plugin.set_actions(elevator=elevator, throttle=throttle, aileron=aileron, flaps=flaps)
        
        # Wait a frame for the action to take effect
        time.sleep(0.1)
        
        new_state = self.plugin.read_state()
        reward = self._calculate_reward(state, new_state, action)
        return new_state, reward

    def _calculate_reward(self, old_state: AiplaneState, new_state: AiplaneState, action: AiplaneAction) -> Reward:
        distance = new_state[0]
        msl = new_state[1]
        lateral_dev = abs(new_state[2])
        vertical_dev = abs(new_state[3])
        vertical_speed = new_state[5]
        
        # Landing reward structure
        reward = 0.0
        
        # Progress towards runway
        reward += (old_state[0] - distance) * 0.01
        
        # Penalty for being off centerline
        reward -= lateral_dev * 0.001
        
        # Penalty for being off glide path
        reward -= vertical_dev * 0.001
        
        # Penalty for excessive sink rate
        reward -= abs(vertical_speed) * 0.0001
        
        # Bonus for successful landing
        if msl < 116 and distance < 100 and lateral_dev < 50:
            reward += 100
            
        # Penalty for crash
        if msl < 114 and (lateral_dev > 100 or abs(vertical_speed) > 8):
            reward -= 50
            
        return reward

    def is_terminal(self, state: AiplaneState) -> bool:
        msl = state[1]
        distance = state[0]
        lateral_dev = abs(state[2])
        
        # Terminal if landed or crashed
        return msl < 116 or distance > 8000 or lateral_dev > 1000

    def actions(self, state: AiplaneState) -> List[AiplaneAction]:
        return self.action_space

    def all_possible_actions(self) -> List[AiplaneAction]:
        return self.action_space

    def is_valid(self, state: AiplaneState, action: AiplaneAction) -> bool:
        return action in self.action_space