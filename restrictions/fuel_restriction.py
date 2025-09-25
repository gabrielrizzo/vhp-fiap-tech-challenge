from typing import List, Tuple, Dict, Any
import math
from core.base_restriction import BaseRestriction

class FuelRestriction(BaseRestriction):
    def __init__(self, max_distance: float = 250.0, fuel_cost_per_km: float = 0.8):
        super().__init__("fuel_restriction")
        self.max_distance = max_distance
        self.fuel_cost_per_km = fuel_cost_per_km
    
    def _calculate_total_distance(self, route: List[Tuple[float, float]]) -> float:
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        n = len(route)
        
        for i in range(n):
            current = route[i]
            next_point = route[(i + 1) % n]
            distance = math.sqrt((current[0] - next_point[0]) ** 2 + (current[1] - next_point[1]) ** 2)
            total_distance += distance
        
        return total_distance
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        total_distance = self._calculate_total_distance(route)
        return total_distance <= self.max_distance
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        total_distance = self._calculate_total_distance(route)
        
        if total_distance <= self.max_distance:
            return 0.0
        
        excess_distance = total_distance - self.max_distance
        penalty = excess_distance * self.fuel_cost_per_km * 10
        
        return penalty
    
    def get_fuel_consumption(self, route: List[Tuple[float, float]]) -> Dict[str, float]:
        total_distance = self._calculate_total_distance(route)
        fuel_cost = total_distance * self.fuel_cost_per_km
        
        return {
            'total_distance': total_distance,
            'fuel_cost': fuel_cost,
            'remaining_fuel_capacity': max(0, self.max_distance - total_distance),
            'fuel_efficiency': total_distance / self.max_distance if self.max_distance > 0 else 0
        }
