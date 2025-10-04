"""
Implementa a restrição de rotas proibidas para o problema do caixeiro viajante.
"""

from typing import List, Tuple, Set, Dict, Any, Optional
from core.base_restriction import BaseRestriction

# Define um tipo para representar uma rota (par de cidades)
Route = Tuple[Tuple[float, float], Tuple[float, float]]


class ForbiddenRoutes(BaseRestriction):
    """
    Implementa a restrição de rotas proibidas.

    Esta restrição simula ruas interditadas, alagamentos, obras, etc.,
    onde não é possível trafegar entre duas cidades.
    """

    def __init__(self, base_distance_penalty: float = 1000.0):
        """
        Inicializa a restrição de rotas proibidas.

        Args:
            base_distance_penalty: Penalidade base para cada rota proibida
        """
        super().__init__("forbidden_routes", 1.0)
        # Armazena as rotas proibidas como um conjunto de tuplas ordenadas
        self._forbidden_routes: Set[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = set()
        self._base_distance_penalty = base_distance_penalty

    def add_forbidden_route(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> None:
        """
        Adiciona uma rota proibida entre duas cidades.
        A ordem das cidades não importa, a rota é considerada bidirecional.

        Args:
            city1: Coordenadas da primeira cidade
            city2: Coordenadas da segunda cidade
        """
        # Ordena as cidades para garantir consistência independente
        # da ordem de entrada
        sorted_cities = sorted([city1, city2])
        route: Tuple[Tuple[float, float], Tuple[float, float]] = (
            sorted_cities[0],
            sorted_cities[1],
        )
        self._forbidden_routes.add(route)

    def is_route_forbidden(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> bool:
        """
        Verifica se uma rota entre duas cidades é proibida.

        Args:
            city1: Coordenadas da primeira cidade
            city2: Coordenadas da segunda cidade

        Returns:
            bool: True se a rota é proibida, False caso contrário
        """
        sorted_cities = sorted([city1, city2])
        route: Tuple[Tuple[float, float], Tuple[float, float]] = (
            sorted_cities[0],
            sorted_cities[1],
        )
        return route in self._forbidden_routes

    def get_all_forbidden_routes(
        self,
    ) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Retorna todas as rotas proibidas.

        Returns:
            Set: Conjunto de rotas proibidas
        """
        return self._forbidden_routes

    def clear_forbidden_routes(self) -> None:
        """
        Remove todas as rotas proibidas.
        """
        self._forbidden_routes.clear()

    def validate_route(
        self,
        route: List[Tuple[float, float]],
        vehicle_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verifica se o caminho não contém rotas proibidas.

        Args:
            route: Lista de coordenadas das cidades no caminho
            vehicle_data: Dados adicionais do veículo (não utilizado)

        Returns:
            bool: True se não contém rotas proibidas, False caso contrário
        """
        n = len(route)
        for i in range(n):
            city1 = route[i]
            city2 = route[(i + 1) % n]
            if self.is_route_forbidden(city1, city2):
                return False
        return True

    def calculate_penalty(
        self,
        route: List[Tuple[float, float]],
        vehicle_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Aplica a restrição de rotas proibidas ao caminho fornecido.

        Args:
            route: Lista de coordenadas das cidades no caminho
            vehicle_data: Dados adicionais do veículo (não utilizado)

        Returns:
            float: Valor da penalidade a ser adicionada ao fitness
        """
        from utils.helper_functions import calculate_distance

        penalty = 0.0
        n = len(route)
        for i in range(n):
            city1 = route[i]
            city2 = route[(i + 1) % n]  # Conecta de volta à primeira cidade

            if self.is_route_forbidden(city1, city2):
                # Aplica uma penalidade proporcional à distância da rota
                route_distance = calculate_distance(city1, city2)
                penalty += self._base_distance_penalty * route_distance

        return penalty

    def get_forbidden_routes_info(
        self, route: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Retorna informações sobre as rotas proibidas no caminho.

        Args:
            route: Lista de coordenadas das cidades no caminho

        Returns:
            Dict[str, Any]: Informações sobre as rotas proibidas
        """
        forbidden_routes_used = []
        n = len(route)

        for i in range(n):
            city1 = route[i]
            city2 = route[(i + 1) % n]

            if self.is_route_forbidden(city1, city2):
                forbidden_routes_used.append((city1, city2))

        return {
            'forbidden_routes_count': len(forbidden_routes_used),
            'forbidden_routes_used': forbidden_routes_used,
            'is_valid': len(forbidden_routes_used) == 0
        }