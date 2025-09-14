from typing import List, Tuple, Dict, Optional
import random
import math

class MultipleVehicles:
    def __init__(self, max_vehicles: int = 5, vehicle_capacity: int = 10):
        """
        Inicializa o sistema de múltiplos veículos (ambulâncias).
        
        Parâmetros:
        - max_vehicles: Número máximo de ambulâncias disponíveis
        - vehicle_capacity: Capacidade máxima de pacientes por ambulância
        """
        self.max_vehicles = max_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_routes: Dict[int, List[Tuple[float, float]]] = {}
        self.vehicle_times: Dict[int, float] = {}
        
    def assign_patients_to_vehicles(self, patients: List[Tuple[float, float]], 
                                  depot: Tuple[float, float]) -> Dict[int, List[Tuple[float, float]]]:
        """
        Distribui os pacientes entre as ambulâncias de forma otimizada.
        
        Parâmetros:
        - patients: Lista de coordenadas dos pacientes
        - depot: Coordenadas do depósito ou hospital base
        
        Retorna:
        - Dict com as rotas de cada ambulância
        """
        # Limpa rotas anteriores
        self.vehicle_routes.clear()
        self.vehicle_times.clear()
        
        # Se não há pacientes, retorna rotas vazias
        if not patients:
            return self.vehicle_routes
            
        # Calcula quantas ambulâncias são necessárias
        n_patients = len(patients)
        n_vehicles_needed = min(
            math.ceil(n_patients / self.vehicle_capacity),
            self.max_vehicles
        )
        
        # Distribui pacientes entre ambulâncias usando algoritmo de clustering simples
        patients_per_vehicle = n_patients // n_vehicles_needed
        remaining_patients = n_patients % n_vehicles_needed
        
        patient_index = 0
        for vehicle_id in range(n_vehicles_needed):
            # Calcula quantos pacientes esta ambulância vai atender
            vehicle_patient_count = patients_per_vehicle
            if vehicle_id < remaining_patients:
                vehicle_patient_count += 1
                
            # Atribui pacientes a esta ambulância
            vehicle_patients = patients[patient_index:patient_index + vehicle_patient_count]
            patient_index += vehicle_patient_count
            
            # Cria rota otimizada para esta ambulância (depósito -> pacientes -> depósito)
            route = self._create_optimized_route(depot, vehicle_patients)
            self.vehicle_routes[vehicle_id] = route
            
            # Calcula tempo total da rota
            self.vehicle_times[vehicle_id] = self._calculate_route_time(route)
            
        return self.vehicle_routes
    
    def _create_optimized_route(self, depot: Tuple[float, float], 
                               patients: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Cria uma rota otimizada para uma ambulância usando algoritmo do vizinho mais próximo.
        """
        if not patients:
            return [depot]
            
        # Começa no depósito
        route = [depot]
        remaining_patients = patients.copy()
        
        # Usa algoritmo do vizinho mais próximo para otimizar a rota
        while remaining_patients:
            current_location = route[-1]
            nearest_patient = min(remaining_patients, 
                                key=lambda p: self._calculate_distance(current_location, p))
            
            route.append(nearest_patient)
            remaining_patients.remove(nearest_patient)
        
        # Retorna ao depósito
        route.append(depot)
        return route
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calcula distância euclidiana entre dois pontos."""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def _calculate_route_time(self, route: List[Tuple[float, float]]) -> float:
        """Calcula o tempo total de uma rota (assumindo velocidade constante)."""
        if len(route) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self._calculate_distance(route[i], route[i + 1])
            
        # Assumindo velocidade de 50 km/h (ajuste conforme necessário)
        speed_kmh = 50.0
        time_hours = total_distance / speed_kmh
        return time_hours * 60  # Retorna em minutos
    
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
    
    def get_vehicle_utilization(self) -> Dict[int, float]:
        """Retorna a utilização de cada ambulância (0-1)."""
        utilization = {}
        for vehicle_id, route in self.vehicle_routes.items():
            # Conta pacientes atendidos (excluindo depósito)
            patients_served = len(route) - 2  # -2 para excluir depósito inicial e final
            utilization[vehicle_id] = patients_served / self.vehicle_capacity
        return utilization

def calculate_fitness_with_multiple_vehicles(
    patients: List[Tuple[float, float]],
    depot: Tuple[float, float],
    multiple_vehicles: MultipleVehicles,
    time_weight: float = 1.0,
    vehicle_count_weight: float = 10.0,
    utilization_weight: float = 5.0
) -> float:
    """
    Calcula o fitness considerando múltiplos veículos.
    
    Parâmetros:
    - patients: Lista de coordenadas dos pacientes
    - depot: Coordenadas do depósito ou hospital base
    - multiple_vehicles: Objeto MultipleVehicles
    - time_weight: Peso para otimização do tempo
    - vehicle_count_weight: Peso para minimizar número de veículos
    - utilization_weight: Peso para maximizar utilização dos veículos
    
    Retorna:
    - float: Fitness total (menor é melhor)
    """
    # Distribui pacientes entre ambulâncias
    vehicle_routes = multiple_vehicles.assign_patients_to_vehicles(patients, depot)
    
    # Calcula métricas
    total_time = multiple_vehicles.get_total_time()
    vehicle_count = multiple_vehicles.get_vehicle_count()
    avg_utilization = sum(multiple_vehicles.get_vehicle_utilization().values()) / max(vehicle_count, 1)
    
    # Fitness = tempo total + penalidade por número de veículos - bônus por utilização
    fitness = (time_weight * total_time + 
              vehicle_count_weight * vehicle_count - 
              utilization_weight * avg_utilization)
    
    return fitness

def optimize_vehicle_assignment(
    patients: List[Tuple[float, float]],
    depot: Tuple[float, float],
    max_vehicles: int = 5,
    vehicle_capacity: int = 10,
    n_iterations: int = 100
) -> Tuple[MultipleVehicles, float]:
    """
    Otimiza a atribuição de pacientes a ambulâncias usando busca local.
    
    Parâmetros:
    - patients: Lista de coordenadas dos pacientes
    - depot: Coordenadas do depósito ou hospital base
    - max_vehicles: Número máximo de ambulâncias
    - vehicle_capacity: Capacidade de cada ambulância
    - n_iterations: Número de iterações para otimização
    
    Retorna:
    - Tuple[MultipleVehicles, fitness]: Melhor configuração encontrada
    """
    best_vehicles = None
    best_fitness = float('inf')
    
    for _ in range(n_iterations):
        # Cria nova configuração de veículos
        vehicles = MultipleVehicles(max_vehicles, vehicle_capacity)
        
        # Embaralha pacientes para diferentes distribuições
        shuffled_patients = patients.copy()
        random.shuffle(shuffled_patients)
        
        # Calcula fitness
        fitness = calculate_fitness_with_multiple_vehicles(
            shuffled_patients, depot, vehicles
        )
        
        # Atualiza melhor solução
        if fitness < best_fitness:
            best_fitness = fitness
            best_vehicles = vehicles
    
    return best_vehicles, best_fitness
