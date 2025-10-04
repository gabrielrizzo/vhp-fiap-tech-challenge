from typing import List, Tuple, Dict, Any
import math
from core.base_restriction import BaseRestriction
from utils.route_utils import RouteUtils


class VehicleCapacityRestriction(BaseRestriction):
    """
    Restrição para capacidade de pacientes em veículos.
    
    Esta restrição define o número máximo de pacientes que uma ambulância pode transportar
    e valida se uma rota pode ser atendida respeitando essa limitação. Quando integrada
    com múltiplos veículos, verifica se é possível distribuir os pacientes entre os
    veículos disponíveis.
    
    Attributes:
        max_patients_per_vehicle (int): Número máximo de pacientes por veículo
    """
    
    def __init__(self, max_patients_per_vehicle: int = 10):
        """
        Inicializa a restrição de capacidade de veículos.
        
        Args:
            max_patients_per_vehicle: Número máximo de pacientes por veículo
        """
        super().__init__("vehicle_capacity_restriction")
        self.max_patients_per_vehicle = max_patients_per_vehicle
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Valida se a rota pode ser atendida respeitando a capacidade máxima de pacientes por veículo.
        
        Para integração com múltiplos veículos, verifica se é possível distribuir os pacientes
        entre os veículos disponíveis respeitando a capacidade de cada um.
        
        Args:
            route: Lista de coordenadas (cidades) da rota
            vehicle_data: Dados do veículo para integração (opcional)
                - vehicle_capacity: Capacidade máxima de pacientes por veículo
                - depot: Coordenadas do depósito/hospital
                - max_vehicles: Número máximo de veículos disponíveis
        
        Returns:
            bool: True se a rota pode ser atendida, False caso contrário
            
        Examples:
            >>> restriction = VehicleCapacityRestriction(max_patients_per_vehicle=5)
            >>> route = [(0, 0), (10, 10), (20, 20)]
            >>> restriction.validate_route(route)
            True
        """
        if not route:
            return True
        
        # Determina a capacidade máxima (prioriza dados do veículo)
        max_capacity = self._get_max_capacity(vehicle_data)
        
        # Extrai pacientes da rota (excluindo depósito se definido)
        patients = self._extract_patients(route, vehicle_data)
        
        # Se há integração com múltiplos veículos, verifica distribuição
        if self._has_multiple_vehicles_integration(vehicle_data):
            return self._validate_multiple_vehicles_distribution(patients, max_capacity, vehicle_data)
        
        # Validação simples: verifica se todos os pacientes cabem em um veículo
        return len(patients) <= max_capacity
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        """
        Calcula penalidade baseada no excesso de pacientes por veículo.
        """
        if not route:
            return 0.0
        
        # Se há dados de veículo específicos, usa a capacidade definida
        if vehicle_data and 'vehicle_capacity' in vehicle_data:
            max_capacity = vehicle_data['vehicle_capacity']
        else:
            max_capacity = self.max_patients_per_vehicle
        
        # Conta apenas os pacientes (exclui depósito se definido)
        depot = vehicle_data.get('depot') if vehicle_data else None
        if depot and len(route) > 1:
            patients = [city for city in route if city != depot]
        else:
            patients = route
        
        patient_count = len(patients)
        
        if patient_count <= max_capacity:
            return 0.0
        
        # Penalidade proporcional ao excesso de pacientes
        excess_patients = patient_count - max_capacity
        penalty = excess_patients * 100  # 100 pontos por paciente em excesso
        
        return penalty
    
    def get_capacity_info(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retorna informações sobre a capacidade do veículo.
        """
        if not route:
            return {
                'patient_count': 0,
                'max_capacity': self.max_patients_per_vehicle,
                'remaining_capacity': self.max_patients_per_vehicle,
                'capacity_utilization': 0.0,
                'is_within_capacity': True
            }
        
        # Se há dados de veículo específicos, usa a capacidade definida
        if vehicle_data and 'vehicle_capacity' in vehicle_data:
            max_capacity = vehicle_data['vehicle_capacity']
        else:
            max_capacity = self.max_patients_per_vehicle
        
        # Conta apenas os pacientes (exclui depósito se definido)
        depot = vehicle_data.get('depot') if vehicle_data else None
        if depot and len(route) > 1:
            patients = [city for city in route if city != depot]
        else:
            patients = route
        
        patient_count = len(patients)
        remaining_capacity = max(0, max_capacity - patient_count)
        capacity_utilization = patient_count / max_capacity if max_capacity > 0 else 0
        
        return {
            'patient_count': patient_count,
            'max_capacity': max_capacity,
            'remaining_capacity': remaining_capacity,
            'capacity_utilization': capacity_utilization,
            'is_within_capacity': patient_count <= max_capacity
        }
    
    def _get_max_capacity(self, vehicle_data: Dict[str, Any] = None) -> int:
        """
        Obtém a capacidade máxima de pacientes por veículo.
        
        Args:
            vehicle_data: Dados do veículo (opcional)
            
        Returns:
            Capacidade máxima de pacientes por veículo
        """
        if vehicle_data and 'vehicle_capacity' in vehicle_data:
            return vehicle_data['vehicle_capacity']
        return self.max_patients_per_vehicle
    
    def _extract_patients(self, route: List[Tuple[float, float]], 
                         vehicle_data: Dict[str, Any] = None) -> List[Tuple[float, float]]:
        """
        Extrai pacientes da rota, excluindo depósito se definido.
        
        Args:
            route: Lista de coordenadas da rota
            vehicle_data: Dados do veículo (opcional)
            
        Returns:
            Lista de coordenadas dos pacientes
        """
        depot = vehicle_data.get('depot') if vehicle_data else None
        return RouteUtils.extract_patients(route, depot)
    
    def _has_multiple_vehicles_integration(self, vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Verifica se há integração com múltiplos veículos.
        
        Args:
            vehicle_data: Dados do veículo (opcional)
            
        Returns:
            True se há integração com múltiplos veículos
        """
        return vehicle_data and 'max_vehicles' in vehicle_data
    
    def _validate_multiple_vehicles_distribution(self, patients: List[Tuple[float, float]], 
                                               max_capacity: int, 
                                               vehicle_data: Dict[str, Any]) -> bool:
        """
        Valida se é possível distribuir pacientes entre múltiplos veículos.
        
        Args:
            patients: Lista de coordenadas dos pacientes
            max_capacity: Capacidade máxima por veículo
            vehicle_data: Dados do veículo
            
        Returns:
            True se é possível distribuir os pacientes
        """
        max_vehicles = vehicle_data['max_vehicles']
        vehicles_needed = math.ceil(len(patients) / max_capacity)
        return vehicles_needed <= max_vehicles
