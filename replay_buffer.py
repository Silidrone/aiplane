import random
from typing import Generic, List, Tuple, TypeVar

S = TypeVar('S')
A = TypeVar('A')


class ReplayBuffer(Generic[S, A]):
    def __init__(self, capacity: int):
        self.buffer: List[Tuple[S, A, float, S, bool]] = []
        self.capacity = capacity
    
    def add(self, state: S, action: A, reward: float, next_state: S, done: bool) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple[S, A, float, S, bool]]:
        batch_size = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        return len(self.buffer)
    
    def get_capacity(self) -> int:
        return self.capacity
    
    def clear(self) -> None:
        self.buffer.clear()