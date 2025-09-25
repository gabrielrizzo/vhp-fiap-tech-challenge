from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

class BaseRestriction(ABC):
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self.enabled = True
    
    @abstractmethod
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        pass
    
    @abstractmethod
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        pass
    
    def is_enabled(self) -> bool:
        return self.enabled
    
    def enable(self):
        self.enabled = True
    
    def disable(self):
        self.enabled = False
    
    def get_weight(self) -> float:
        return self.weight
    
    def set_weight(self, weight: float):
        self.weight = weight
