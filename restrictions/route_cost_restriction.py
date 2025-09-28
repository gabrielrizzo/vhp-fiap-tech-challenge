"""
    Classe para implementação de custo de rotas. 
"""

from typing import List, Tuple, Dict, Any
from core.base_restriction import BaseRestriction
class RouteCostRestriction(BaseRestriction):
    """
        Implementação do custo para rota das cidades. Deixamos em aberto para poder
        ser a somatória de pedágio ou qualquer outro custo que seja necessário.
        Não é necessário definir o custo de todas as rotas, somente aquelas que temos algo definido
    """
    def __init__(self, cities_locations: List[Tuple[float, float]],
            route_cost_dict: Dict,
            weight: float = 1):
        super().__init__("route_cost_restriction", weight)
        self.cities_locations = cities_locations
        self.route_cost_dict = route_cost_dict

    def validate_route(
            self,
            route: List[Tuple[float, float]],
            vehicle_data: Dict[str, Any] = None
    ) -> bool:
        return True

    def calculate_penalty(
            self,
            route: List[Tuple[float, float]],
            vehicle_data: Dict[str, Any] = None
    ) -> float:
        """
        Calcula fitness usando custos de rotas diretamente do route_cost_dict.
        
        Args:
            path: Lista de coordenadas das cidades na solução
            
        Returns:
            float: custo_total_rotas
        """
        total_route_cost = 0

        for i, current_city in enumerate(route):  # i é o índice, current_city é a cidade
            next_city = route[(i + 1) % len(route)]  # Próxima cidade usando o índice

            # Busca custo da rota no dicionário
            route_key = (current_city, next_city)
            reverse_key = (next_city, current_city)

            # print(route)
            if route_key in self.route_cost_dict:
                total_route_cost += self.route_cost_dict[route_key]
            elif reverse_key in self.route_cost_dict:
                total_route_cost += self.route_cost_dict[reverse_key]
            # Se não encontrar, custo é 0 (rota gratuita)

        return total_route_cost

    def get_route_cost(self, route: List[Tuple[float, float]]) -> str:
        """
            Retorna o quanto de custo de rota foi consumido
        """

        total_route_cost = self.calculate_penalty(route)

        return {
            "total_route_cost": total_route_cost,
        }
    