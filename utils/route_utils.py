"""
Utilitários para operações comuns com rotas e veículos.
"""
from typing import List, Tuple, Dict, Any
import math
from functools import lru_cache


class RouteUtils:
    """Classe utilitária para operações comuns com rotas."""
    
    @staticmethod
    def extract_patients(route: List[Tuple[float, float]], 
                        depot: Tuple[float, float] = None) -> List[Tuple[float, float]]:
        """
        Extrai pacientes da rota, excluindo depósito se definido.
        
        Args:
            route: Lista de coordenadas (cidades) da rota
            depot: Coordenadas do depósito/hospital (opcional)
            
        Returns:
            Lista de coordenadas dos pacientes (excluindo depósito)
        """
        if not depot or len(route) <= 1:
            return route
        return [city for city in route if city != depot]
    
    @staticmethod
    @lru_cache(maxsize=10000)
    def calculate_distance(point1: Tuple[float, float],
        point2:  Tuple[float, float], use_geographic=False) -> float:
        """
        Calcula distância euclidiana entre dois pontos (com cache LRU otimizado).
        
        Args:
            point1: Primeiro ponto (x, y)
            point2: Segundo ponto (x, y)
            
        Returns:
            Distância euclidiana entre os pontos
        """
        if use_geographic:
            # Para coordenadas geográficas (lat, lon)
            return RouteUtils.haversine_distance(
                point1[0], point1[1],
                point2[0], point2[1]
            )
        else:
            # Para pixels (ATT48)
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calcula distância real em km entre coordenadas geográficas"""
        R = 6371  # Raio da Terra em km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + \
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c  # Retorna em KM
    
    @staticmethod
    def calculate_route_distance(route: List[Tuple[float, float]], use_geographic=False) -> float:
        """
        Calcula distância total de uma rota (otimizado com cache).
        
        Args:
            route: Lista de coordenadas da rota
            
        Returns:
            Distância total da rota
        """
        if len(route) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += RouteUtils.calculate_distance(route[i], route[i + 1], use_geographic)
        
        return total_distance
    
    @staticmethod
    @lru_cache(maxsize=2000)
    def calculate_route_distance_cached(route_tuple: Tuple[Tuple[float, float], ...]) -> float:
        """
        Calcula distância total de uma rota com cache LRU (versão otimizada).
        
        Args:
            route_tuple: Tupla de coordenadas da rota (deve ser convertida para tupla)
            
        Returns:
            Distância total da rota
        """
        if len(route_tuple) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(route_tuple) - 1):
            total_distance += RouteUtils.calculate_distance(route_tuple[i], route_tuple[i + 1])
        
        return total_distance
    
    @staticmethod
    def calculate_route_time(route: List[Tuple[float, float]], 
                           speed_kmh: float = 50.0) -> float:
        """
        Calcula tempo total de uma rota.
        
        Args:
            route: Lista de coordenadas da rota
            speed_kmh: Velocidade em km/h (padrão: 50)
            
        Returns:
            Tempo total em minutos
        """
        if len(route) < 2:
            return 0.0
        
        # Usa distância cacheada por rota completa quando possível
        try:
            route_tuple = tuple(route)
            total_distance = RouteUtils.calculate_route_distance_cached(route_tuple)
        except TypeError:
            # Fallback para lista não hashable ou dados inconsistentes
            total_distance = RouteUtils.calculate_route_distance(route)
        time_hours = total_distance / speed_kmh
        return time_hours * 60  # Retorna em minutos
    
    @staticmethod
    def get_vehicle_data_template(vehicle_capacity: int, 
                                depot: Tuple[float, float] = None,
                                max_vehicles: int = None) -> Dict[str, Any]:
        """
        Cria template de dados do veículo para integração.
        
        Args:
            vehicle_capacity: Capacidade de pacientes por veículo
            depot: Coordenadas do depósito (opcional)
            max_vehicles: Número máximo de veículos (opcional)
            
        Returns:
            Dicionário com dados do veículo
        """
        data = {'vehicle_capacity': vehicle_capacity}
        if depot is not None:
            data['depot'] = depot
        if max_vehicles is not None:
            data['max_vehicles'] = max_vehicles
        return data
