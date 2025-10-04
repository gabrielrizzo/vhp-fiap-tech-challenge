from typing import List, Tuple, Dict, Any
import math
from core.base_restriction import BaseRestriction
from utils.route_utils import RouteUtils

class FuelRestriction(BaseRestriction):
    def __init__(
        self,
        max_distance: float = 250.0,
        fuel_cost_per_km: float = 0.8,
        fuel_cost_limit: float = None,
        pixel_to_km_factor: float = 0.02
    ):
        super().__init__("fuel_restriction")
        self.max_distance = max_distance
        self.fuel_cost_per_km = fuel_cost_per_km
        self.fuel_cost_limit = fuel_cost_limit
        self.pixel_to_km_factor = pixel_to_km_factor

    def _calculate_total_distance(self, route: List[Tuple[float, float]]) -> float:
        """
        Calcula distância total da rota usando RouteUtils otimizado.
        
        Args:
            route: Lista de coordenadas da rota
            
        Returns:
            Distância total da rota
        """
        return RouteUtils.calculate_route_distance(route) * self.pixel_to_km_factor
    
    def validate_route(self, route: List[Tuple[float, float]], vehicle_data: Dict[str, Any] = None) -> bool:
        total_distance = self._calculate_total_distance(route)
        fuel_cost = total_distance * self.fuel_cost_per_km
        # Verifica a existencia do limite primeiro, para podermos habilitar ou não o limite de custo
        # Caso não tenha o limite, é retornado true
        is_fuel_cost_valid = not(self.fuel_cost_limit) or fuel_cost <= self.fuel_cost_limit

        return total_distance <= self.max_distance and is_fuel_cost_valid

    def calculate_penalty(
            self,
            route: List[Tuple[float, float]],
            vehicle_data: Dict[str, Any] = None
    ) -> float:
        total_distance = self._calculate_total_distance(route)
        fuel_cost = total_distance * self.fuel_cost_per_km
        penalty = 0

        if total_distance > self.max_distance:
            excess_distance = total_distance - self.max_distance
            penalty = excess_distance * self.fuel_cost_per_km * 10

        # Adiciona uma penalidade caso o custo seja maior que o limite
        if self.fuel_cost_limit and fuel_cost > self.fuel_cost_limit:
            penalty = penalty + (fuel_cost * 3)

        # Caso não tenha passado a cota de combustível e nem a de distância, retorna penalidade 0
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
