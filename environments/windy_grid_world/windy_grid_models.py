import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from torch_model import TorchModel
from environments.windy_grid_world.windy_grid_constants import GRID_HEIGHT, GRID_WIDTH
from environments.windy_grid_world.windy_grid_world import GridPosition, UP, DOWN, LEFT, RIGHT, ACTIONS

class WindyGridWorldQNet(TorchModel):
    def __init__(self, input_size: int, hidden_size: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.output(x)

def feature_extractor(state: GridPosition, action: Tuple[int, int], device=None) -> torch.Tensor:
    device = torch.device("cpu")
    
    row, col = state
    action_idx = ACTIONS.index(action)
    total_actions = len(ACTIONS)
    feature_size = GRID_HEIGHT * GRID_WIDTH * total_actions
    features = torch.zeros(feature_size, device=device)
    index = (row * GRID_WIDTH + col) * total_actions + action_idx
    features[index] = 1.0
    features = features.view(1, -1)
    return features