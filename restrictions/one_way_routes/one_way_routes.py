from typing import List, Tuple, Dict, Any, Set
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from genetic_algorithm import calculate_distance, calculate_fitness
from restrictions.restriction_interface import RestrictionInterface


class OneWayRoutes(RestrictionInterface):
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
        # Armazena as rotas unidirecionais como um conjunto de tuplas ordenadas (origem, destino)
        # Onde a direção permitida é de origem -> destino
        self._one_way_routes: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
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
            A direção contrária (destination -> origin) será considerada contramão.
        """
        # Armazena a rota unidirecional como um par ordenado (origem, destino)
        self._one_way_routes.add((origin, destination))

    def is_wrong_way(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> bool:
        """
        Verifica se uma rota está na contramão.
        
        Args:
            origin: Coordenadas da cidade de origem
            destination: Coordenadas da cidade de destino
            
        Returns:
            bool: True se a rota está na contramão, False caso contrário
        """
        # Se existe uma rota unidirecional na direção contrária, então está na contramão
        return (destination, origin) in self._one_way_routes

    def get_all_one_way_routes(self) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
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

    # Implementação dos métodos da interface
    def fitness_restriction(self, path: List[Tuple[float, float]]) -> float:
        """
        Aplica a restrição de rotas unidirecionais ao caminho fornecido.
        
        Args:
            path: Lista de coordenadas das cidades no caminho
            
        Returns:
            float: Valor da penalidade a ser adicionada ao fitness
        """
        penalty = 0.0
        n = len(path)
        for i in range(n):
            city1 = path[i]
            city2 = path[(i + 1) % n]  # Conecta de volta à primeira cidade

            if self.is_wrong_way(city1, city2):
                # Aplica uma penalidade proporcional à distância da rota em contramão
                route_distance = calculate_distance(city1, city2)
                penalty += self._base_distance_penalty * route_distance

        return penalty
    
    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """
        Verifica se o caminho não contém rotas na contramão.
        
        Args:
            path: Lista de coordenadas das cidades no caminho
            
        Returns:
            bool: True se o caminho não contém rotas na contramão, False caso contrário
        """
        n = len(path)
        for i in range(n):
            city1 = path[i]
            city2 = path[(i + 1) % n]
            if self.is_wrong_way(city1, city2):
                return False
        return True
    
    def get_name(self) -> str:
        """
        Retorna o nome da restrição.
        
        Returns:
            str: Nome da restrição
        """
        return "Rotas Unidirecionais"
    
    def get_description(self) -> str:
        """
        Retorna a descrição da restrição.
        
        Returns:
            str: Descrição detalhada da restrição
        """
        return "Restrição que simula vias de mão única, onde só é possível trafegar em uma direção específica entre duas cidades."
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros configuráveis da restrição.
        
        Returns:
            Dict[str, Any]: Dicionário com os parâmetros e seus valores atuais
        """
        return {
            "base_distance_penalty": self._base_distance_penalty,
            "one_way_routes_count": len(self._one_way_routes)
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configura os parâmetros da restrição.
        
        Args:
            parameters: Dicionário com os parâmetros a serem configurados
        """
        if "base_distance_penalty" in parameters:
            self._base_distance_penalty = parameters["base_distance_penalty"]


def calculate_fitness_with_one_way_restrictions(
    path: List[Tuple[float, float]],
    one_way_routes: OneWayRoutes,
) -> float:
    """
    Calcula o fitness de um caminho considerando rotas unidirecionais.

    Parâmetros:
    - path: Lista de coordenadas das cidades no caminho
    - one_way_routes: Objeto OneWayRoutes contendo as rotas unidirecionais

    Retorna:
    - float: Distância total do caminho + penalidades por rotas na contramão
    """
    # Calcula a distância normal do caminho
    base_fitness = calculate_fitness(path)

    # Adiciona penalidades para rotas na contramão
    penalty = one_way_routes.fitness_restriction(path)

    return base_fitness + penalty
