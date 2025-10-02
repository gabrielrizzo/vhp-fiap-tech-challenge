from typing import List, Tuple, Dict, Any
from core.base_restriction import BaseRestriction

class AmbulancePatientRestriction(BaseRestriction):
    """
        Limita a capacidade de pacientes por ambulância
    """
    def __init__(self, max_patients: int = 10, vehicle_data: Dict[str, Any] = None):
        super().__init__("ambulance_patient_restriction")
        self.vehicle_data = vehicle_data or {}
        self.max_patients = max_patients
    
    def _calculate_total_patients(self, route: List[Tuple[float, float]]) -> int:
        if self.vehicle_data and 'city_patients' in self.vehicle_data:
            total_patients = 0
            for city in route:
                city_key = f"{city[0]}_{city[1]}"
                patients = self.vehicle_data['city_patients'].get(city_key, 0)  # Padrão: 0 paciente por cidade
                total_patients += patients
            return total_patients
        
        # Sem dados específicos, retorna sem pacientes
        return 0
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        total_patients = self._calculate_total_patients(route)
        return total_patients <= self.max_patients
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        total_patients = self._calculate_total_patients(route)
        
        if total_patients <= self.max_patients:
            return 0.0
        
        excess_patients = total_patients - self.max_patients
        penalty = excess_patients * 1000  # Penalidade alta por exceder capacidade de pacientes
        
        return penalty
    
    def get_capacity_info(self, route: List[Tuple[float, float]]) -> Dict[str, Any]:
        total_patients = self._calculate_total_patients(route)
        
        return {
            'total_patients': total_patients,
            'max_patients': self.max_patients,
            'remaining_capacity': max(0, self.max_patients - total_patients),
            'capacity_utilization': total_patients / self.max_patients if self.max_patients > 0 else 0,
            'cities_count': len(route)
        }
