import torch
import torch.nn as nn

from torch_model import TorchModel
from typing import Tuple

# State: (distance, msl, lateral_dev, vertical_dev, heading_dev, vertical_speed, pitch, bank, airspeed)
AiplaneState = Tuple[float, float, float, float, float, float, float, float, float]
# Action: (elevator, throttle, aileron, flaps)
AiplaneAction = Tuple[float, float, float, float]

class AiplaneQNet(TorchModel):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

def feature_extractor(state: AiplaneState, action: AiplaneAction, device=torch.device("cpu")) -> torch.Tensor:
    # Normalize state features
    distance, msl, lateral_dev, vertical_dev, heading_dev, vertical_speed, pitch, bank, airspeed = state
    elevator, throttle, aileron, flaps = action
    
    # Normalization ranges - distance/MSL: optimal->1, deviations: 0->0  
    # Distance: 0 (runway) = 1, 6000 = -1
    norm_distance = max(-1.0, min(1.0, 1 - 2 * distance / 6000))
    # MSL: runway level (114.028) = 1, higher = lower values
    norm_msl = max(-1.0, min(1.0, 1 - 2 * (msl - 114.028) / (450 - 114.028)))
    # Deviations: 0 = 0, max deviation = ±1
    norm_lateral = max(-1.0, min(1.0, lateral_dev / 750))
    norm_vertical = max(-1.0, min(1.0, vertical_dev / 50))
    norm_heading = max(-1.0, min(1.0, heading_dev / 10))
    norm_vspeed = max(-1.0, min(1.0, vertical_speed / 10))
    norm_pitch = max(-1.0, min(1.0, pitch / 20))
    norm_bank = max(-1.0, min(1.0, bank / 25))
    # Airspeed: 0 = 0, 70 = 1
    norm_airspeed = max(0.0, min(1.0, airspeed / 70))
    
    # Normalize actions - [-1, 1] for elevator/aileron, [0, 1] for throttle/flaps
    norm_elevator = max(-1.0, min(1.0, 2 * (elevator + 1) / 2 - 1))
    norm_throttle = max(0.0, min(1.0, throttle))
    norm_aileron = max(-1.0, min(1.0, 2 * (aileron + 1) / 2 - 1))
    norm_flaps = max(0.0, min(1.0, flaps))
    
    features = [
        norm_distance, norm_msl, norm_lateral, norm_vertical,
        norm_heading, norm_vspeed, norm_pitch, norm_bank, norm_airspeed,
        norm_elevator, norm_throttle, norm_aileron, norm_flaps
    ]
    
    return torch.tensor([features], dtype=torch.float32, device=device)