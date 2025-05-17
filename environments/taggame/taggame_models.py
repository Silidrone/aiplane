import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_model import TorchModel
from environments.taggame.taggame_constants import (HIDDEN_SIZE, MAX_DISTANCE, MAX_VELOCITY, MAX_X, MAX_Y,
                        N_OF_EPISODES)
from environments.taggame.tag_game import TagGameAction, TagGameState


class TagGameQNet(TorchModel):
    def __init__(self, input_size: int, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)


def feature_extractor(state: TagGameState, action: TagGameAction, device=None) -> torch.Tensor:
    device = torch.device("cpu")  # Force CPU for stability
        
    my_pos, my_vel, tag_pos, tag_vel, is_tagged = state
    action_x, action_y = action
    
    features = []
    
    norm_my_pos_x = my_pos[0] / MAX_X
    norm_my_pos_y = my_pos[1] / MAX_Y
    norm_my_vel_x = my_vel[0] / MAX_VELOCITY
    norm_my_vel_y = my_vel[1] / MAX_VELOCITY
    norm_tag_pos_x = tag_pos[0] / MAX_X
    norm_tag_pos_y = tag_pos[1] / MAX_Y
    norm_tag_vel_x = tag_vel[0] / MAX_VELOCITY
    norm_tag_vel_y = tag_vel[1] / MAX_VELOCITY
    
    norm_action_x = action_x / MAX_VELOCITY
    norm_action_y = action_y / MAX_VELOCITY
    
    dx = my_pos[0] - tag_pos[0]
    dy = my_pos[1] - tag_pos[1]
    distance = math.sqrt(dx * dx + dy * dy) / math.sqrt(MAX_X * MAX_X + MAX_Y * MAX_Y)
    
    features.append(norm_my_pos_x)
    features.append(norm_my_pos_y)
    features.append(norm_my_vel_x)
    features.append(norm_my_vel_y)
    features.append(norm_tag_pos_x)
    features.append(norm_tag_pos_y)
    features.append(norm_tag_vel_x)
    features.append(norm_tag_vel_y)
    features.append(norm_action_x)
    features.append(norm_action_y)
    features.append(distance)
    features.append(1.0 if is_tagged else 0.0)
    features.append(1.0)
    
    return torch.tensor(features, dtype=torch.float32, device=device).view(1, -1)