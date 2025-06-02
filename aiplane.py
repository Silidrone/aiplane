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
        # Discrete action space for SARSA - REDUCED FOR SPEED
        elevators = [-0.3, 0.0, 0.3]
        throttles = [0.3, 0.5, 0.7]
        ailerons = [-0.2, 0.0, 0.2]
        flaps = [0.0, 1.0]
        
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
        # time.sleep(0.1)  # REMOVED FOR SPEED
        
        new_state = self.plugin.read_state()
        reward = self._calculate_reward(state, new_state, action)
        return new_state, reward

    def _calculate_reward(self, old_state: AiplaneState, new_state: AiplaneState, action: AiplaneAction) -> Reward:
        distance, msl, lateral_dev, vertical_dev, heading_dev, vertical_speed, pitch, bank, airspeed, rec_flaps, rec_throttle, rec_pitch = new_state
        elevator, throttle, aileron, flaps = action
        
        # Landing reward structure
        reward = 0.0
        
        # Progress towards runway
        reward += (old_state[0] - distance) * 0.01
        
        # Penalty for deviations (aligned with normalized features centered at 0)
        reward -= abs(lateral_dev) * 0.001
        reward -= abs(vertical_dev) * 0.001
        reward -= abs(heading_dev) * 0.005
        reward -= abs(vertical_speed) * 0.0001
        reward -= abs(pitch) * 0.0005
        reward -= abs(bank) * 0.0005
        
        # Cheat rewards: bonus for following recommendations
        reward += 0.1 * (1.0 - abs(flaps - rec_flaps))  # Flap compliance bonus
        reward += 0.1 * (1.0 - abs(throttle - rec_throttle))  # Throttle compliance bonus
        reward += 0.05 * (1.0 - abs(pitch - rec_pitch) / 10.0)  # Pitch compliance bonus
        
        # Bonus for successful landing
        if msl < 116 and distance < 100 and abs(lateral_dev) < 50:
            reward += 100
            
        # Penalty for crash
        if msl < 116 and (abs(lateral_dev) > 100 or abs(vertical_speed) > 8):
            reward -= 50
            
        return reward

    def is_terminal(self, state: AiplaneState) -> bool:
        distance, msl, lateral_dev, vertical_dev, heading_dev, vertical_speed, pitch, bank, airspeed, _, _, _ = state
        
        # Terminal conditions for faster convergence (relaxed)
        return (
            msl < 116 or  # Landed/crashed
            distance > 12000 or  # Too far from runway
            abs(lateral_dev) > 2000 or  # Too far off centerline
            abs(vertical_dev) > 200 or  # Too far off glide path
            abs(heading_dev) > 90 or  # Wrong heading
            abs(vertical_speed) > 30 or  # Excessive sink/climb rate
            abs(pitch) > 45 or  # Excessive pitch
            abs(bank) > 60 or  # Excessive bank
            airspeed < 20 or airspeed > 150  # Stall or overspeed
        )

    def actions(self, state: AiplaneState) -> List[AiplaneAction]:
        return self.action_space

    def all_possible_actions(self) -> List[AiplaneAction]:
        return self.action_space

    def is_valid(self, state: AiplaneState, action: AiplaneAction) -> bool:
        return action in self.action_space