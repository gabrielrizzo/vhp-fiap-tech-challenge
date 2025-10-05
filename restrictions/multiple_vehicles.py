from typing import List, Tuple, Dict, Any
import math
from core.base_restriction import BaseRestriction
from utils.route_utils import RouteUtils


class MultipleVehiclesRestriction(BaseRestriction):
    """
    Restrição de múltiplos veículos (frota) para TSP médico (ambulâncias).
    
    Objetivo
    - Distribuir pacientes entre veículos respeitando capacidade, gerando para cada veículo
      uma rota simples (vizinho mais próximo) e métricas úteis (tempo, utilização, veículos usados).
    
    Parâmetros principais
    - `max_vehicles`: máximo de veículos disponíveis
    - `depot`: depósito/hospital (início/fim das rotas)
    - `vehicle_capacity`: capacidade por veículo
    
    Integração
    - Exponibiliza `vehicle_data` para outras restrições (p.ex., capacidade), incluindo:
      `vehicle_capacity`, `depot`, `max_vehicles`, e, quando rota é fornecida, também
      `vehicles_used` e `unserved_patients`.
    """
    
    def __init__(self, max_vehicles: int = 5, depot: Tuple[float, float] = None, 
                 vehicle_capacity: int = 1):
        """
        Inicializa a restrição de múltiplos veículos.
        
        Args:
            max_vehicles: Número máximo de ambulâncias disponíveis
            depot: Coordenadas (cidade) do depósito ou hospital base
            vehicle_capacity: Capacidade de pacientes por veículo
        """
        super().__init__("multiple_vehicles_restriction")
        self.max_vehicles = max_vehicles
        self.depot = depot
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_routes: Dict[int, List[Tuple[float, float]]] = {}
        self.vehicle_times: Dict[int, float] = {}
        
    def _get_effective_params(self, vehicle_data: Dict[str, Any] = None) -> Tuple[int, Tuple[float, float], int]:
        """Retorna (vehicle_capacity, depot, max_vehicles) considerando overrides de vehicle_data."""
        if not vehicle_data:
            return self.vehicle_capacity, self.depot, self.max_vehicles
        capacity = vehicle_data.get('vehicle_capacity', self.vehicle_capacity)
        depot = vehicle_data.get('depot', self.depot)
        max_vehicles = vehicle_data.get('max_vehicles', self.max_vehicles)
        return capacity, depot, max_vehicles
        
    def set_depot(self, depot: Tuple[float, float]):
        """Define qual cidade é o depósito/hospital base"""
        self.depot = depot
    
    def set_vehicle_capacity(self, capacity: int):
        """Define a capacidade de pacientes por veículo"""
        self.vehicle_capacity = capacity
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """
        Calcula distância euclidiana entre dois pontos (cidades).
        
        Args:
            point1: Primeiro ponto (x, y)
            point2: Segundo ponto (x, y)
            
        Returns:
            Distância euclidiana entre os pontos
        """
        return RouteUtils.calculate_distance(point1, point2)
    
    def _create_optimized_route(self, depot: Tuple[float, float], 
                               patients: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Cria uma rota simples por vizinho mais próximo: depot -> pacientes -> depot.
        """
        if not patients:
            return [depot]
            
        # Começa no depósito ou hospital base
        route = [depot]
        remaining_patients = patients.copy()
        
        # Usa algoritmo do vizinho mais próximo para otimizar a rota
        while remaining_patients:
            current_location = route[-1]
            nearest_patient = min(remaining_patients, 
                                key=lambda p: self._calculate_distance(current_location, p))
            
            route.append(nearest_patient)
            remaining_patients.remove(nearest_patient)
        
        # Retorna ao depósito ou hospital base
        route.append(depot)
        return route
    
    def _calculate_route_time(self, route: List[Tuple[float, float]], speed_kmh: float = 50.0) -> float:
        """
        Calcula tempo total (min) de uma rota.
        
        Args:
            route: Lista de coordenadas da rota
            speed_kmh: Velocidade em km/h (padrão: 50)
            
        Returns:
            Tempo total da rota em minutos
        """
        return RouteUtils.calculate_route_time(route, speed_kmh)
    
    def _distribute_patients_to_vehicles(self, patients: List[Tuple[float, float]], 
                                        depot: Tuple[float, float],
                                        vehicle_capacity: int,
                                        max_vehicles: int) -> Dict[int, List[Tuple[float, float]]]:
        """
        Distribui pacientes sequencialmente por veículo até capacidade ou limite de frota,
        criando rotas individuais por vizinho mais próximo.
        
        Args:
            patients: Lista de coordenadas (cidades) dos pacientes
            depot: Coordenadas (cidade) do depósito ou hospital base
        
        Returns:
            Dict com as rotas (cidades) de cada ambulância
            
        Observação: distribuição gulosa/sequencial suficiente para penalização/relatórios.
        """
        # Limpa rotas anteriores
        self.vehicle_routes.clear()
        self.vehicle_times.clear()
        
        # Se não há pacientes, retorna rotas vazias
        if not patients:
            return self.vehicle_routes
            
        # Calcula quantas ambulâncias seriam necessárias e quantas podem ser usadas
        n_patients = len(patients)
        vehicles_needed = math.ceil(n_patients / vehicle_capacity) if vehicle_capacity > 0 else 0
        n_vehicles_to_use = min(vehicles_needed, max_vehicles)
        
        # Distribui pacientes respeitando a capacidade máxima por veículo
        patient_index = 0
        for vehicle_id in range(n_vehicles_to_use):
            # Calcula quantos pacientes esta ambulância pode atender
            remaining_patients = n_patients - patient_index
            vehicle_patient_count = min(vehicle_capacity, remaining_patients)
            
            # Atribui pacientes a esta ambulância
            vehicle_patients = patients[patient_index:patient_index + vehicle_patient_count]
            patient_index += vehicle_patient_count
            
            # Cria rota otimizada para esta ambulância (depósito -> pacientes -> depósito)
            route = self._create_optimized_route(depot, vehicle_patients)
            self.vehicle_routes[vehicle_id] = route
            
            # Calcula tempo total da rota
            self.vehicle_times[vehicle_id] = self._calculate_route_time(route)
            
        return self.vehicle_routes
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        """
        Valida se a rota pode ser atendida com o número máximo de veículos disponíveis.
        """
        # Lê parâmetros efetivos (permitindo overrides de outra restrição)
        eff_capacity, eff_depot, eff_max_vehicles = self._get_effective_params(vehicle_data)

        if not eff_depot or not route:
            return True  # Se não tiver depósito definido, aceita qualquer rota
        
        # Remove o depósito da rota para contar apenas os pacientes
        patients = [city for city in route if city != eff_depot]
        
        if not patients:
            return True  # Se não há pacientes, a rota é válida
        
        # Calcula quantas ambulâncias seriam necessárias
        n_patients = len(patients)
        vehicles_needed = math.ceil(n_patients / eff_capacity) if eff_capacity > 0 else float('inf')
        
        # Verifica se o número de veículos necessários não excede o máximo disponível
        return vehicles_needed <= eff_max_vehicles
    
    def calculate_penalty(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> float:
        """
        Calcula penalidade baseada no número de veículos necessários e eficiência das rotas.
        """
        eff_capacity, eff_depot, eff_max_vehicles = self._get_effective_params(vehicle_data)
        if not eff_depot or not route:
            return 0.0
        
        # Remove o depósito da rota para contar apenas os pacientes
        patients = [city for city in route if city != eff_depot]
        
        if not patients:
            return 0.0  # Se não há pacientes, não há penalidade
        
        # Distribui pacientes entre veículos
        vehicle_routes = self._distribute_patients_to_vehicles(patients, eff_depot, eff_capacity, eff_max_vehicles)
        
        # Calcula métricas
        total_time = self.get_total_time()
        vehicle_count = self.get_vehicle_count()
        avg_utilization = self.get_average_utilization()
        
        # Penalidade por pacientes não atendidos (se exceder capacidade total disponível)
        total_capacity_available = eff_capacity * eff_max_vehicles
        unserved_patients = max(0, len(patients) - total_capacity_available)
        
        # Penalidade baseada no número de veículos e eficiência
        vehicle_penalty = vehicle_count * 100  # Penalidade por usar mais veículos
        time_penalty = total_time * 0.1  # Penalidade por tempo total
        utilization_bonus = avg_utilization * 50  # Bônus por boa utilização
        unserved_penalty = unserved_patients * 500  # Penalidade alta por paciente sem atendimento
        
        penalty = vehicle_penalty + time_penalty - utilization_bonus + unserved_penalty
        
        return max(0.0, penalty)  # Não permite penalidade negativa
    
    def get_total_time(self) -> float:
        """Retorna o tempo total de todas as ambulâncias (tempo da ambulância mais lenta)."""
        if not self.vehicle_times:
            return 0.0
        return max(self.vehicle_times.values())
    
    def get_average_time(self) -> float:
        """Retorna o tempo médio das ambulâncias."""
        if not self.vehicle_times:
            return 0.0
        return sum(self.vehicle_times.values()) / len(self.vehicle_times)
    
    def get_vehicle_count(self) -> int:
        """Retorna o número de ambulâncias utilizadas."""
        return len(self.vehicle_routes)
    
    def get_average_utilization(self) -> float:
        """Retorna a utilização média das ambulâncias (0-1)."""
        if not self.vehicle_routes:
            return 0.0
        
        total_utilization = 0.0
        for vehicle_id, route in self.vehicle_routes.items():
            # Conta pacientes atendidos (excluindo depósito)
            patients_served = len(route) - 2  # -2 para excluir depósito inicial e final
            utilization = patients_served / self.vehicle_capacity  # Capacidade configurável
            total_utilization += utilization
        
        return total_utilization / len(self.vehicle_routes)
    
    def get_vehicle_utilization(self) -> Dict[int, float]:
        """Retorna a utilização de cada ambulância (0-1)."""
        utilization = {}
        for vehicle_id, route in self.vehicle_routes.items():
            # Conta pacientes atendidos (excluindo depósito)
            patients_served = len(route) - 2  # -2 para excluir depósito inicial e final
            utilization[vehicle_id] = patients_served / self.vehicle_capacity  # Capacidade configurável
        return utilization
    
    def get_multiple_vehicles_info(self, route: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Informações sobre a restrição de múltiplos veículos
        """
        if not route or not self.depot:
            return {"error": "Empty route or no depot defined"}
        
        # Remove o depósito da rota para contar apenas os pacientes
        patients = [city for city in route if city != self.depot]
        
        if not patients:
            return {
                'depot': self.depot,
                'patients_count': 0,
                'vehicles_needed': 0,
                'max_vehicles': self.max_vehicles,
                'can_handle_route': True
            }
        
        # Calcula veículos necessários e pacientes não atendidos
        vehicles_needed = math.ceil(len(patients) / self.vehicle_capacity) if self.vehicle_capacity > 0 else float('inf')
        total_capacity_available = self.vehicle_capacity * self.max_vehicles
        unserved_patients = max(0, len(patients) - total_capacity_available)
        
        # Distribui pacientes entre veículos (limitado ao máximo disponível)
        vehicle_routes = self._distribute_patients_to_vehicles(
            patients, self.depot, self.vehicle_capacity, self.max_vehicles
        )
        
        return {
            'depot': self.depot,
            'patients_count': len(patients),
            'vehicles_needed': vehicles_needed,
            'max_vehicles': self.max_vehicles,
            'can_handle_route': vehicles_needed <= self.max_vehicles,
            'total_time': self.get_total_time(),
            'average_time': self.get_average_time(),
            'average_utilization': self.get_average_utilization(),
            'vehicle_utilization': self.get_vehicle_utilization(),
            'vehicle_capacity': self.vehicle_capacity,
            'unserved_patients': unserved_patients
        }
    
    def get_vehicle_data_for_capacity_restriction(self, route: List[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Retorna dados do veículo para integração com VehicleCapacityRestriction.
        
        Returns:
            Dicionário com dados necessários para integração:
                - vehicle_capacity: Capacidade de pacientes por veículo
                - depot: Coordenadas do depósito/hospital
                - max_vehicles: Número máximo de veículos disponíveis
                - vehicles_used: Número efetivo de veículos usados para a rota (se rota fornecida)
                - unserved_patients: Pacientes não atendidos dado o limite (se rota fornecida)
        """
        data = RouteUtils.get_vehicle_data_template(
            vehicle_capacity=self.vehicle_capacity,
            depot=self.depot,
            max_vehicles=self.max_vehicles
        )
        
        # Se uma rota for fornecida, calcula vehicles_used e unserved_patients
        if route and self.depot:
            # Remove o depósito da rota para contar apenas os pacientes
            patients = [city for city in route if city != self.depot]
            
            if patients:
                # Distribui respeitando limites atuais
                self._distribute_patients_to_vehicles(
                    patients,
                    self.depot,
                    self.vehicle_capacity,
                    self.max_vehicles
                )
                data['vehicles_used'] = self.get_vehicle_count()
                total_capacity = self.vehicle_capacity * self.max_vehicles
                data['unserved_patients'] = max(0, len(patients) - total_capacity)
            else:
                data['vehicles_used'] = 0
                data['unserved_patients'] = 0
        
        return data
