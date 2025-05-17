import os
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, Tuple, TypeVar, Union
import torch
import torch.nn as nn
import torch.optim as optim
import json

from mdp import MDP, Action, State
from function_approximator import FunctionApproximator
from torch_model import TorchModel

Return = float

class ValueStrategy(Generic[State, Action], ABC):
    @abstractmethod
    def initialize(self, mdp: MDP[State, Action]) -> None:
        pass
    
    @abstractmethod
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        pass
    
    @abstractmethod
    def Q(self, state: State, action: Action) -> float:
        pass
    
    @abstractmethod
    def update(self, state: State, action: Action, target_q: float) -> None:
        pass
    
    def save(self, path: str) -> None:
        raise NotImplementedError("Save not implemented for this ValueStrategy")
    
    def load(self, path: str) -> None:
        raise NotImplementedError("Load not implemented for this ValueStrategy")


class TabularValueStrategy(ValueStrategy[State, Action]):
    def __init__(self, step_size: float = 0.1):
        self._Q: Dict[Tuple[State, Action], Return] = {}
        self._mdp = None
        self._strict = False
        self._step_size = step_size
    
    def set_strict_mode(self, strict: bool) -> None:
        self._strict = strict
    
    def set_step_size(self, step_size: float) -> None:
        self._step_size = step_size
    
    def initialize(self, mdp: MDP[State, Action]) -> None:
        self._mdp = mdp
        
        for state in mdp.states():
            for action in mdp.actions(state):
                self._Q[(state, action)] = 0.0
        
        for state in mdp.terminal_states():
            for action in mdp.actions(state):
                self._Q[(state, action)] = 0.0
    
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        if self._mdp is None:
            raise RuntimeError("TabularValueStrategy not initialized with an MDP")
        
        max_value = float('-inf')
        best_action = None
        
        for action in self._mdp.actions(state):
            value = self.Q(state, action)
            if value > max_value:
                max_value = value
                best_action = action
        
        return best_action, max_value
    
    def Q(self, state: State, action: Action) -> float:
        if (state, action) in self._Q:
            return self._Q[(state, action)]
        
        if self._strict:
            raise KeyError("Invalid state-action pair provided for the Q-value function")
        
        return 0.0
    
    def update(self, state: State, action: Action, target_q: float) -> None:
        current_q = self.Q(state, action)
        updated_q = current_q + self._step_size * (target_q - current_q)
        self.set_q(state, action, updated_q)
    
    def set_q(self, state: State, action: Action, value: float) -> None:
        self._Q[(state, action)] = value
    
    def get_q_table(self) -> Dict[Tuple[State, Action], Return]:
        return self._Q
    
    def save(self, path: str) -> None:
        serializable_q = {str((s, a)): v for (s, a), v in self._Q.items()}
        
        try:
            with open(path, 'w') as f:
                json.dump(serializable_q, f)
        except Exception as e:
            raise IOError(f"Failed to save Q-table: {e}")
    
    def load(self, path: str) -> None:
        try:
            with open(path, 'r') as f:
                serialized_q = json.load(f)
            
            print(f"Loaded Q-table with {len(serialized_q)} entries")
            
        except Exception as e:
            raise IOError(f"Failed to load Q-table: {e}")


class ApproximationValueStrategy(ValueStrategy[State, Action]):
    def __init__(self, approximator: FunctionApproximator[State, Action] = None, 
                 step_size: float = 0.1):
        self._approximator = approximator
        self._mdp = None
        self._step_size = step_size
    
    def initialize(self, mdp: MDP[State, Action]) -> None:
        if mdp is None or self._approximator is None:
            raise ValueError("Both MDP and FunctionApproximator must be non-null")
        
        self._mdp = mdp
    
    def initialize_with_approximator(self, mdp: MDP[State, Action], 
                                    approximator: FunctionApproximator[State, Action]) -> None:
        self._mdp = mdp
        self._approximator = approximator
    
    def set_approximator(self, approximator: FunctionApproximator[State, Action]) -> None:
        self._approximator = approximator
    
    def set_step_size(self, step_size: float) -> None:
        self._step_size = step_size
    
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        if self._approximator is None or self._mdp is None:
            raise RuntimeError("ApproximationValueStrategy not properly initialized")
        
        best_action = None
        best_value = float('-inf')
        
        for action in self._mdp.actions(state):
            if not self._mdp.is_valid(state, action):
                continue
            
            value = self.Q(state, action)
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value
    
    def Q(self, state: State, action: Action) -> float:
        if self._approximator is None:
            raise RuntimeError("ApproximationValueStrategy: No approximator set")
        
        return self._approximator.predict(state, action)
    
    def update(self, state: State, action: Action, target_q: float) -> None:
        if self._approximator is None:
            raise RuntimeError("ApproximationValueStrategy: No approximator set")
        
        current_q = self.Q(state, action)
        error = target_q - current_q
        self._approximator.update(state, action, error, self._step_size)
    
    @property
    def approximator(self) -> FunctionApproximator[State, Action]:
        return self._approximator
    
    def save(self, path: str) -> None:
        try:
            weights = self._approximator.get_weights()
            with open(path, 'w') as f:
                json.dump(weights, f)
        except Exception as e:
            raise IOError(f"Failed to save weights: {e}")
    
    def load(self, path: str) -> None:
        try:
            with open(path, 'r') as f:
                weights = json.load(f)
            self._approximator.set_weights(weights)
        except Exception as e:
            raise IOError(f"Failed to load weights: {e}")


class TorchValueStrategy(ValueStrategy[State, Action]):
    def __init__(self, network: TorchModel, 
                 feature_extractor: Callable[[State, Action], torch.Tensor],
                 step_size: float = 0.01):
        self.device = torch.device("cpu")  # Force CPU for stability
        
        self.q_network = network
        self.feature_extractor = feature_extractor
        self.q_network.to(self.device)
        self.optimizer = optim.Adam(network.parameters(), lr=step_size)
        self._mdp = None
        self._step_size = step_size
    
    def initialize(self, mdp: MDP[State, Action]) -> None:
        self._mdp = mdp
    
    def get_best_action(self, state: State) -> Tuple[Action, Return]:
        if self._mdp is None:
            raise RuntimeError("TorchValueStrategy not initialized with an MDP")
        
        best_action = None
        best_value = float('-inf')
        
        for action in self._mdp.actions(state):
            if self._mdp.is_valid(state, action):
                value = self.Q(state, action)
                if value > best_value:
                    best_value = value
                    best_action = action
        
        return best_action, best_value
    
    def Q(self, state: State, action: Action) -> float:
        with torch.no_grad():
            state_action = self.feature_extractor(state, action, device=self.device)
            return self.q_network(state_action).item()
    
    def update(self, state: State, action: Action, target_q: float) -> None:
        state_action = self.feature_extractor(state, action, device=self.device)
        current_q = self.q_network(state_action)
        target = torch.tensor([[target_q]], dtype=current_q.dtype, device=self.device)
        
        loss = nn.functional.mse_loss(current_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def save(self, path: str) -> None:
        try:
            self.q_network.to('cpu')
            torch.save(self.q_network.state_dict(), path)
            self.q_network.to(self.device)
        except Exception as e:
            self.q_network.to(self.device)
            raise IOError(f"Failed to save model: {e}")
    
    def load(self, path: str) -> None:
        try:
            self.q_network.load_state_dict(torch.load(path, map_location=self.device))
            self.q_network.to(self.device)
        except Exception as e:
            raise IOError(f"Failed to load model: {e}")