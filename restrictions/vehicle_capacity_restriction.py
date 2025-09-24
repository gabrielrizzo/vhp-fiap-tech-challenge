from typing import List, Tuple, Dict, Any
from core.base_restriction import BaseRestriction

class VehicleCapacityRestriction(BaseRestriction):
    def __init__(self, max_capacity: int = 10, delivery_weight_per_city: float = 1.0):
        super().__init__("vehicle_capacity_restriction")
        self.max_capacity = max_capacity
        self.delivery_weight_per_city = delivery_weight_per_city
    
    def _calculate_total_weight(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        if vehicle_data and 'city_weights' in vehicle_data:
            total_weight = 0.0
            for city in route:
                city_key = f"{city[0]}_{city[1]}"
                weight = vehicle_data['city_weights'].get(city_key, self.delivery_weight_per_city)
                total_weight += weight
            return total_weight
        
        return len(route) * self.delivery_weight_per_city
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        total_weight = self._calculate_total_weight(route, vehicle_data)
        return total_weight <= self.max_capacity
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        total_weight = self._calculate_total_weight(route, vehicle_data)
        
        if total_weight <= self.max_capacity:
            return 0.0
        
        excess_weight = total_weight - self.max_capacity
        penalty = excess_weight * 100
        
        return penalty
    
    def get_capacity_info(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> Dict[str, float]:
        total_weight = self._calculate_total_weight(route, vehicle_data)
        
        return {
            'total_weight': total_weight,
            'max_capacity': self.max_capacity,
            'remaining_capacity': max(0, self.max_capacity - total_weight),
            'capacity_utilization': total_weight / self.max_capacity if self.max_capacity > 0 else 0,
            'cities_count': len(route)
        }
