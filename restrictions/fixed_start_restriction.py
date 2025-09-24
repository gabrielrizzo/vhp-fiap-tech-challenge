from typing import List, Tuple, Dict, Any
from core.base_restriction import BaseRestriction
 
class FixedStartRestriction(BaseRestriction):
    def __init__(self, hospital_location: Tuple[float, float] = None):
        super().__init__("fixed_start_restriction")
        # Se não especificar, usa a primeira cidade como hospital
        self.hospital_location = hospital_location
    
    def set_hospital_location(self, location: Tuple[float, float]):
        """Define qual cidade é o hospital base"""
        self.hospital_location = location
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Valida se a rota começa no hospital (base)
        """
        if not route or not self.hospital_location:
            return True  # Se não tiver hospital definido, aceita qualquer rota
        
        # Verifica se primeira cidade da rota é o hospital
        first_city = route[0]
        return first_city == self.hospital_location
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        """
        Penaliza rotas que não começam no hospital
        """
        if self.validate_route(route, vehicle_data):
            return 0.0
        
        # Penalidade alta por não começar no hospital
        return 5000.0  # Penalidade muito alta para forçar começar no hospital
    
    def get_hospital_info(self, route: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Informações sobre a restrição de hospital
        """
        if not route:
            return {"error": "Empty route"}
        
        return {
            'hospital_location': self.hospital_location,
            'route_start': route[0],
            'starts_at_hospital': self.validate_route(route),
            'first_city_is_correct': route[0] == self.hospital_location if self.hospital_location else False
        }