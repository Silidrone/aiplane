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
        elevators = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        throttles = [0.0, 0.33, 0.66, 1.0]
        ailerons = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        flaps = [0.0, 0.33, 0.66, 1.0]
        
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
        
        new_state = self.plugin.read_state()
        reward = self._calculate_reward(state, new_state, action)
        return new_state, reward

    def _calculate_reward(self, old_state: AiplaneState, new_state: AiplaneState, action: AiplaneAction) -> Reward:
        distance, msl, lateral_dev, vertical_dev, vertical_speed, pitch, bank, airspeed = new_state
        elevator, throttle, aileron, flaps = action

        # Successful landing
        if msl < 116 and distance < 100 and abs(lateral_dev) < 50:
            return 100
            
        # Penalty for terminal conditions (crash/violation)
        if self.is_terminal(new_state):
            return -100
            
        return 0.1

    def is_terminal(self, state: AiplaneState) -> bool:
        distance, msl, lateral_dev, vertical_dev, vertical_speed, pitch, bank, airspeed = state
        
        return (
            msl < 116 or  # Landed/crashed
            distance > 6000 or  # Too far from runway (stricter: was 12000)
            abs(lateral_dev) > 500 or  # Too far off centerline (stricter: was 2000)
            abs(vertical_dev) > 120 or  # Too far off glide path (stricter: was 200)
            abs(pitch) > 35 or  # Excessive pitch (stricter: was 45)
            abs(bank) > 40 or  # Excessive bank (stricter: was 60)
            airspeed < 25 or airspeed > 120  # Stall or overspeed (stricter: was 20-150)
        )

    def actions(self, state: AiplaneState) -> List[AiplaneAction]:
        return self.action_space

    def all_possible_actions(self) -> List[AiplaneAction]:
        return self.action_space

    def is_valid(self, state: AiplaneState, action: AiplaneAction) -> bool:
        return action in self.action_space