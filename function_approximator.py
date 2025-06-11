"""
Author: Muhamed Cicak
"""

import random
from abc import ABC, abstractmethod
from typing import Callable, Generic, List, TypeVar

import numpy as np

S = TypeVar('S')
A = TypeVar('A')

class FunctionApproximator(Generic[S, A], ABC):
    @abstractmethod
    def predict(self, state: S, action: A) -> float:
        pass
    
    @abstractmethod
    def gradient(self, state: S, action: A) -> List[float]:
        pass
    
    @abstractmethod
    def update(self, state: S, action: A, error: float, step_size: float) -> None:
        pass
    
    @abstractmethod
    def get_weights(self) -> List[float]:
        pass
    
    @abstractmethod
    def set_weights(self, weights: List[float]) -> None:
        pass


class LinearFunctionApproximator(FunctionApproximator[S, A]):
    def __init__(self, feature_dim: int, feature_extractor: Callable[[S, A], List[float]]):
        self.weights = np.zeros(feature_dim)
        self.feature_extractor = feature_extractor
        
        for i in range(feature_dim):
            self.weights[i] = random.uniform(-0.1, 0.1)
    
    def predict(self, state: S, action: A) -> float:
        features = np.array(self.feature_extractor(state, action))
        return float(np.dot(self.weights, features))
    
    def gradient(self, state: S, action: A) -> List[float]:
        return self.feature_extractor(state, action)
    
    def update(self, state: S, action: A, error: float, step_size: float) -> None:
        features = np.array(self.feature_extractor(state, action))
        self.weights += step_size * error * features
    
    def get_weights(self) -> List[float]:
        return self.weights.tolist()
    
    def set_weights(self, weights: List[float]) -> None:
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected weights of length {len(self.weights)}, got {len(weights)}")
        self.weights = np.array(weights)


class NeuralNetworkFunctionApproximator(FunctionApproximator[S, A]):
    def __init__(self, feature_extractor: Callable[[S, A], List[float]], 
                 architecture: List[int]):
        if len(architecture) < 2:
            raise ValueError("Network architecture must have at least input and output layers")
        
        self.feature_extractor = feature_extractor
        self.layer_sizes = architecture
        
        self.weights = []
        
        for i in range(len(architecture) - 1):
            input_size = architecture[i]
            output_size = architecture[i + 1]
            
            for _ in range(output_size):
                self.weights.append(random.uniform(-0.1, 0.1))
                
            for _ in range(output_size * input_size):
                self.weights.append(random.uniform(-0.1, 0.1))
    
    def _forward(self, features: List[float]) -> List[List[float]]:
        activations = [features]
        weight_idx = 0
        
        for l in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[l]
            output_size = self.layer_sizes[l + 1]
            
            layer_output = [0.0] * output_size
            
            for j in range(output_size):
                layer_output[j] = self.weights[weight_idx]
                weight_idx += 1
                
                for i in range(input_size):
                    layer_output[j] += self.weights[weight_idx] * activations[-1][i]
                    weight_idx += 1
                
                if l < len(self.layer_sizes) - 2:
                    layer_output[j] = max(0.0, layer_output[j])
            
            activations.append(layer_output)
        
        return activations
    
    def predict(self, state: S, action: A) -> float:
        features = self.feature_extractor(state, action)
        activations = self._forward(features)
        return activations[-1][0]
    
    def gradient(self, state: S, action: A) -> List[float]:
        features = self.feature_extractor(state, action)
        activations = self._forward(features)
        
        gradients = [0.0] * len(self.weights)
        deltas = [[] for _ in range(len(self.layer_sizes))]
        
        deltas[-1] = [1.0]
        
        for l in range(len(self.layer_sizes) - 2, -1, -1):
            current_size = self.layer_sizes[l]
            next_size = self.layer_sizes[l + 1]
            deltas[l] = [0.0] * current_size
            
            weight_idx = 0
            for i in range(l):
                weight_idx += self.layer_sizes[i + 1] * (self.layer_sizes[i] + 1)
            weight_idx += next_size
            
            for i in range(current_size):
                for j in range(next_size):
                    weight = self.weights[weight_idx + j * current_size + i]
                    deltas[l][i] += weight * deltas[l + 1][j]
                
                if l > 0:
                    if activations[l][i] <= 0:
                        deltas[l][i] = 0.0
        
        weight_idx = 0
        for l in range(len(self.layer_sizes) - 1):
            input_size = self.layer_sizes[l]
            output_size = self.layer_sizes[l + 1]
            
            for j in range(output_size):
                gradients[weight_idx] = deltas[l + 1][j]
                weight_idx += 1
                
                for i in range(input_size):
                    gradients[weight_idx] = deltas[l + 1][j] * activations[l][i]
                    weight_idx += 1
        
        return gradients
    
    def update(self, state: S, action: A, error: float, step_size: float) -> None:
        grads = self.gradient(state, action)
        for i in range(len(self.weights)):
            self.weights[i] += step_size * error * grads[i]
    
    def get_weights(self) -> List[float]:
        return self.weights
    
    def set_weights(self, weights: List[float]) -> None:
        if len(weights) != len(self.weights):
            raise ValueError(f"Expected weights of length {len(self.weights)}, got {len(weights)}")
        self.weights = weights