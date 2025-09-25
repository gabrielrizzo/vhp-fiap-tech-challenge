from typing import List, Tuple, Dict, Any
from core.base_restriction import BaseRestriction

class RestrictionManager:
    def __init__(self):
        self.restrictions: List[BaseRestriction] = []
        self.base_fitness_weight = 1.0
    
    def add_restriction(self, restriction: BaseRestriction):
        self.restrictions.append(restriction)
    
    def remove_restriction(self, restriction_name: str):
        self.restrictions = [r for r in self.restrictions if r.name != restriction_name]
    
    def get_restriction(self, name: str) -> BaseRestriction:
        for restriction in self.restrictions:
            if restriction.name == name:
                return restriction
        return None
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        for restriction in self.restrictions:
            if restriction.is_enabled() and not restriction.validate_route(route, vehicle_data):
                return False
        return True
    
    def calculate_fitness_with_restrictions(self, route: List[Tuple[float, float]], 
                                          base_fitness: float, 
                                          vehicle_data: Dict[str, Any] = None) -> float:
        total_penalty = 0.0
        
        for restriction in self.restrictions:
            if restriction.is_enabled():
                penalty = restriction.calculate_penalty(route, vehicle_data)
                total_penalty += penalty * restriction.get_weight()
        
        return (base_fitness * self.base_fitness_weight) + total_penalty
    
    def get_active_restrictions(self) -> List[str]:
        return [r.name for r in self.restrictions if r.is_enabled()]
    
    def enable_restriction(self, name: str):
        restriction = self.get_restriction(name)
        if restriction:
            restriction.enable()
    
    def disable_restriction(self, name: str):
        restriction = self.get_restriction(name)
        if restriction:
            restriction.disable()
    
    def set_restriction_weight(self, name: str, weight: float):
        restriction = self.get_restriction(name)
        if restriction:
            restriction.set_weight(weight)
    
    def get_violation_summary(self, route: List[Tuple[float, float]], 
                            vehicle_data: Dict[str, Any] = None) -> Dict[str, Any]:
        summary = {
            'valid_route': True,
            'violations': [],
            'total_penalty': 0.0
        }
        
        for restriction in self.restrictions:
            if restriction.is_enabled():
                is_valid = restriction.validate_route(route, vehicle_data)
                penalty = restriction.calculate_penalty(route, vehicle_data)
                
                if not is_valid or penalty > 0:
                    summary['violations'].append({
                        'restriction': restriction.name,
                        'valid': is_valid,
                        'penalty': penalty
                    })
                    summary['valid_route'] = False
                
                summary['total_penalty'] += penalty * restriction.get_weight()
        
        return summary
