from typing import List, Tuple, Dict, Any
import math
from core.base_restriction import BaseRestriction
from utils.route_utils import RouteUtils


class VehicleCapacityRestriction(BaseRestriction):
    """
    Restrição de capacidade por veículo / ambulância (TSP médico).
    
    Objetivo:
    - Garantir que a quantidade de pacientes por veículo não exceda a capacidade.
    - Com frota (múltiplos veículos), validar se a demanda total cabe na capacidade agregada
      e/ou nos sinais já calculados (`vehicles_used`, `unserved_patients`).
    
    Integração (`vehicle_data` esperado):
    - `vehicle_capacity` (int): capacidade por veículo
    - `max_vehicles` (int): quantidade máxima de veículos
    - `depot` (tuple): coordenadas do depósito (excluído da contagem)
    - `vehicles_used` (int, opcional): veículos efetivamente usados
    - `unserved_patients` (int, opcional): pacientes não atendidos
    
    Observação:
    - Sem `vehicle_data`, a restrição não bloqueia a rota (retorna True), pois não há
      contexto suficiente para validar frota/capacidade.
    """
    
    def __init__(self, max_patients_per_vehicle: int = 10):
        """
        Inicializa a restrição de capacidade de veículos.
        """
        super().__init__("vehicle_capacity_restriction")
        self.max_patients_per_vehicle = max_patients_per_vehicle
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Valida a rota contra a capacidade disponível.
        
        - Sem `vehicle_data`: retorna True (não há contexto para validar capacidade).
        - Com frota (`max_vehicles` presente): usa distribuição mínima necessária (ceil)
          ou sinais diretos (`vehicles_used`/`unserved_patients`).
        - Sem frota: valida contra a capacidade de um único veículo.
        """
        if not route:
            return True
        
        # Determina a capacidade máxima (prioriza dados do veículo)
        max_capacity = self._get_max_capacity(vehicle_data)
        
        # Extrai pacientes da rota (excluindo depósito se definido)
        patients = self._extract_patients(route, vehicle_data)

        # Se não há vehicle_data, não há contexto para frota; não aplica restrição de capacidade de forma conservadora
        if vehicle_data is None:
            return True

        # Se há integração com múltiplos veículos, verifica distribuição
        if self._has_multiple_vehicles_integration(vehicle_data):
            # Se já houver um cálculo de vehicles_used/unserved_patients, respeita-o
            vehicles_used = vehicle_data.get('vehicles_used') if vehicle_data else None
            unserved_patients = vehicle_data.get('unserved_patients') if vehicle_data else None
            if vehicles_used is not None or unserved_patients is not None:
                if unserved_patients is not None and unserved_patients > 0:
                    return False
                # Caso contrário, validamos também contra o limite de max_vehicles
                max_vehicles = vehicle_data.get('max_vehicles', float('inf'))
                return vehicles_used is None or vehicles_used <= max_vehicles
            
            # Fallback: calcula distribuição mínima possível dado max_vehicles
            return self._validate_multiple_vehicles_distribution(patients, max_capacity, vehicle_data)
        
        # Validação simples: verifica se todos os pacientes cabem em um veículo
        return len(patients) <= max_capacity
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        """
        Calcula penalidade baseada no excesso de pacientes, considerando frota quando aplicável.
        """
        if not route:
            return 0.0
        
        # Capacidade máxima por veículo (override por vehicle_data)
        max_capacity = vehicle_data['vehicle_capacity'] if vehicle_data and 'vehicle_capacity' in vehicle_data else self.max_patients_per_vehicle
        
        # Extrai pacientes (exclui depósito quando definido)
        depot = vehicle_data.get('depot') if vehicle_data else None
        if depot and len(route) > 1:
            patients = [city for city in route if city != depot]
        else:
            patients = route
        
        patient_count = len(patients)
        penalty = 0.0
        
        # Integração com múltiplos veículos quando disponível
        if vehicle_data and 'max_vehicles' in vehicle_data:
            max_vehicles = vehicle_data.get('max_vehicles', 0)
            total_capacity = max_capacity * max_vehicles
            
            # Se houver `unserved_patients` calculado, prioriza esse sinal
            unserved_patients = vehicle_data.get('unserved_patients')
            if isinstance(unserved_patients, int) and unserved_patients > 0:
                penalty += unserved_patients * 300  # penalidade por paciente não atendido
            else:
                # Caso não haja o campo, calcula excesso acima da capacidade total disponível
                if patient_count > total_capacity:
                    excess_total = patient_count - total_capacity
                    penalty += excess_total * 300
            
            # Opcional: pequena penalidade se aproximando do limite (evita saturação total)
            if total_capacity > 0:
                utilization = patient_count / total_capacity
                if utilization > 0.9:
                    penalty += (utilization - 0.9) * 50
        else:
            # Sem integração de múltiplos veículos: penaliza excesso acima da capacidade do veículo único
            if patient_count > max_capacity:
                excess_patients = patient_count - max_capacity
                penalty += excess_patients * 100
        
        return penalty
    
    def get_capacity_info(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Retorna um resumo didático sobre capacidade e utilização.
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
        Retorna a capacidade efetiva do veículo (prioriza `vehicle_data`).
        """
        if vehicle_data and 'vehicle_capacity' in vehicle_data:
            return vehicle_data['vehicle_capacity']
        return self.max_patients_per_vehicle
    
    def _extract_patients(self, route: List[Tuple[float, float]], 
                         vehicle_data: Dict[str, Any] = None) -> List[Tuple[float, float]]:
        """
        Extrai pacientes da rota, excluindo o depósito quando informado.
        """
        depot = vehicle_data.get('depot') if vehicle_data else None
        return RouteUtils.extract_patients(route, depot)
    
    def _has_multiple_vehicles_integration(self, vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Indica se há contexto de frota disponível via `vehicle_data`.
        """
        return vehicle_data and 'max_vehicles' in vehicle_data
    
    def _validate_multiple_vehicles_distribution(self, patients: List[Tuple[float, float]], 
                                               max_capacity: int, 
                                               vehicle_data: Dict[str, Any]) -> bool:
        """
        Verifica se a demanda cabe na frota mínima necessária
        (ceil(len(patients)/capacidade)) respeitando `max_vehicles`.
        """
        max_vehicles = vehicle_data['max_vehicles']
        vehicles_needed = math.ceil(len(patients) / max_capacity)
        return vehicles_needed <= max_vehicles
