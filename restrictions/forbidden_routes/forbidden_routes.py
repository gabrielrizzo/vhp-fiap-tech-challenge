from typing import List, Tuple, Set, Dict, Any
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from genetic_algorithm import calculate_distance, calculate_fitness
from restrictions.restriction_interface import RestrictionInterface

# Define um tipo para representar uma rota (par de cidades)
Route = Tuple[Tuple[float, float], Tuple[float, float]]


class ForbiddenRoutes(RestrictionInterface):
    def __init__(self, base_distance_penalty: float = 1000.0):
        # Armazena as rotas proibidas como um conjunto de tuplas ordenadas
        self._forbidden_routes: Set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
        self._base_distance_penalty = base_distance_penalty

    def add_forbidden_route(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> None:
        """
        Adiciona uma rota proibida entre duas cidades.
        A ordem das cidades não importa, a rota é considerada bidirecional.
        """
        # Ordena as cidades para garantir consistência independente da ordem de entrada
        sorted_cities = sorted([city1, city2])
        route: Tuple[Tuple[float, float], Tuple[float, float]] = (sorted_cities[0], sorted_cities[1])
        self._forbidden_routes.add(route)

    def is_route_forbidden(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> bool:
        """
        Verifica se uma rota entre duas cidades é proibida.
        """
        sorted_cities = sorted([city1, city2])
        route: Tuple[Tuple[float, float], Tuple[float, float]] = (sorted_cities[0], sorted_cities[1])
        return route in self._forbidden_routes

    def get_all_forbidden_routes(self) -> Set[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Retorna todas as rotas proibidas.
        """
        return self._forbidden_routes

    def clear_forbidden_routes(self) -> None:
        """
        Remove todas as rotas proibidas.
        """
        self._forbidden_routes.clear()

    # Implementação dos métodos da interface
    def fitness_restriction(self, path: List[Tuple[float, float]]) -> float:
        """
        Aplica a restrição de rotas proibidas ao caminho fornecido.
        
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

            if self.is_route_forbidden(city1, city2):
                # Aplica uma penalidade proporcional à distância da rota proibida
                route_distance = calculate_distance(city1, city2)
                penalty += self._base_distance_penalty * route_distance

        return penalty
    
    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """
        Verifica se o caminho não contém rotas proibidas.
        
        Args:
            path: Lista de coordenadas das cidades no caminho
            
        Returns:
            bool: True se o caminho não contém rotas proibidas, False caso contrário
        """
        n = len(path)
        for i in range(n):
            city1 = path[i]
            city2 = path[(i + 1) % n]
            if self.is_route_forbidden(city1, city2):
                return False
        return True
    
    def get_name(self) -> str:
        """
        Retorna o nome da restrição.
        
        Returns:
            str: Nome da restrição
        """
        return "Rotas Proibidas"
    
    def get_description(self) -> str:
        """
        Retorna a descrição da restrição.
        
        Returns:
            str: Descrição detalhada da restrição
        """
        return "Restrição que impede o uso de certas rotas entre cidades, simulando ruas interditadas, alagamentos, obras, etc."
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros configuráveis da restrição.
        
        Returns:
            Dict[str, Any]: Dicionário com os parâmetros e seus valores atuais
        """
        return {
            "base_distance_penalty": self._base_distance_penalty,
            "forbidden_routes_count": len(self._forbidden_routes)
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configura os parâmetros da restrição.
        
        Args:
            parameters: Dicionário com os parâmetros a serem configurados
        """
        if "base_distance_penalty" in parameters:
            self._base_distance_penalty = parameters["base_distance_penalty"]


def calculate_fitness_with_restrictions(
    path: List[Tuple[float, float]],
    forbidden_routes: ForbiddenRoutes,
) -> float:
    """
    Calcula o fitness de um caminho considerando rotas proibidas.

    Parâmetros:
    - path: Lista de coordenadas das cidades no caminho
    - forbidden_routes: Objeto ForbiddenRoutes contendo as rotas proibidas

    Retorna:
    - float: Distância total do caminho + penalidades por rotas proibidas
    """
    # Calcula a distância normal do caminho
    base_fitness = calculate_fitness(path)

    # Adiciona penalidades para rotas proibidas
    penalty = forbidden_routes.fitness_restriction(path)

    return base_fitness + penalty
