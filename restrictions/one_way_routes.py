"""
Implementa a restrição de rotas unidirecionais (mão única).
"""

from typing import List, Tuple, Dict, Any, Set, Optional
from core.base_restriction import BaseRestriction


class OneWayRoutes(BaseRestriction):
    """
    Implementa a restrição de rotas unidirecionais (mão única).

    Esta restrição simula vias de mão única, onde só é possível trafegar
    em uma direção específica entre duas cidades.
    """

    def __init__(self, base_distance_penalty: float = 2000.0):
        """
        Inicializa a restrição de rotas unidirecionais.

        Args:
            base_distance_penalty: Penalidade base para cada rota em contramão
        """
        super().__init__("one_way_routes", 1.0)
        # Armazena as rotas unidirecionais como um conjunto de tuplas
        # (origem, destino) onde a direção permitida é de origem -> destino
        self._one_way_routes: Set[
            Tuple[Tuple[float, float], Tuple[float, float]]
        ] = set()
        self._base_distance_penalty = base_distance_penalty

    def add_one_way_route(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> None:
        """
        Adiciona uma rota unidirecional entre duas cidades.

        Args:
            origin: Coordenadas da cidade de origem
            destination: Coordenadas da cidade de destino

        Note:
            A rota só poderá ser percorrida na direção origin -> destination.
            A direção contrária será considerada contramão.
        """
        # Armazena a rota unidirecional como um par ordenado (origem, destino)
        self._one_way_routes.add((origin, destination))

    def is_wrong_way(
        self, origin: Tuple[float, float], destination: Tuple[float, float]
    ) -> bool:
        """
        Verifica se uma rota está na contramão.

        Args:
            origin: Coordenadas da cidade de origem
            destination: Coordenadas da cidade de destino

        Returns:
            bool: True se a rota está na contramão, False caso contrário
        """
        # Se existe uma rota unidirecional na direção contrária, está na contramão
        return (destination, origin) in self._one_way_routes

    def get_all_one_way_routes(
        self,
    ) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Retorna todas as rotas unidirecionais.

        Returns:
            Set: Conjunto de rotas unidirecionais
        """
        return self._one_way_routes

    def clear_one_way_routes(self) -> None:
        """
        Remove todas as rotas unidirecionais.
        """
        self._one_way_routes.clear()

    def validate_route(
        self,
        route: List[Tuple[float, float]],
        vehicle_data: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Verifica se o caminho não contém rotas na contramão.

        Args:
            route: Lista de coordenadas das cidades no caminho
            vehicle_data: Dados adicionais do veículo (não utilizado)

        Returns:
            bool: True se não contém rotas na contramão, False caso contrário
        """
        n = len(route)
        for i in range(n):
            city1 = route[i]
            city2 = route[(i + 1) % n]
            if self.is_wrong_way(city1, city2):
                return False
        return True

    def calculate_penalty(
        self,
        route: List[Tuple[float, float]],
        vehicle_data: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Aplica a restrição de rotas unidirecionais ao caminho fornecido.

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

            if self.is_wrong_way(city1, city2):
                # Aplica uma penalidade proporcional à distância da rota
                route_distance = calculate_distance(city1, city2)
                penalty += self._base_distance_penalty * route_distance

        return penalty

    def get_wrong_way_routes(
        self, route: List[Tuple[float, float]]
    ) -> Dict[str, Any]:
        """
        Retorna informações sobre as rotas em contramão no caminho.

        Args:
            route: Lista de coordenadas das cidades no caminho

        Returns:
            Dict[str, Any]: Informações sobre as rotas em contramão
        """
        wrong_way_routes = []
        n = len(route)

        for i in range(n):
            city1 = route[i]
            city2 = route[(i + 1) % n]

            if self.is_wrong_way(city1, city2):
                wrong_way_routes.append((city1, city2))

        return {
            'wrong_way_count': len(wrong_way_routes),
            'wrong_way_routes': wrong_way_routes,
            'is_valid': len(wrong_way_routes) == 0
        }