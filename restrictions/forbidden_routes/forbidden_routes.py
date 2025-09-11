from typing import List, Tuple, Set, TypeAlias

# Define um tipo para representar uma rota (par de cidades)
Route: TypeAlias = Tuple[Tuple[float, float], Tuple[float, float]]


class ForbiddenRoutes:
    def __init__(self):
        # Armazena as rotas proibidas como um conjunto de tuplas ordenadas
        self._forbidden_routes: Set[Route] = set()

    def add_forbidden_route(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> None:
        """
        Adiciona uma rota proibida entre duas cidades.
        A ordem das cidades não importa, a rota é considerada bidirecional.
        """
        # Ordena as cidades para garantir consistência independente da ordem de entrada
        sorted_cities = sorted([city1, city2])
        route: Route = (sorted_cities[0], sorted_cities[1])
        self._forbidden_routes.add(route)

    def is_route_forbidden(
        self, city1: Tuple[float, float], city2: Tuple[float, float]
    ) -> bool:
        """
        Verifica se uma rota entre duas cidades é proibida.
        """
        sorted_cities = sorted([city1, city2])
        route: Route = (sorted_cities[0], sorted_cities[1])
        return route in self._forbidden_routes

    def get_all_forbidden_routes(self) -> Set[Route]:
        """
        Retorna todas as rotas proibidas.
        """
        return self._forbidden_routes

    def clear_forbidden_routes(self) -> None:
        """
        Remove todas as rotas proibidas.
        """
        self._forbidden_routes.clear()

def calculate_fitness_with_restrictions(
    path: List[Tuple[float, float]],
    forbidden_routes: ForbiddenRoutes,
    base_distance_penalty: float = 1000.0,
) -> float:
    """
    Calcula o fitness de um caminho considerando rotas proibidas.

    Parâmetros:
    - path: Lista de coordenadas das cidades no caminho
    - forbidden_routes: Objeto ForbiddenRoutes contendo as rotas proibidas
    - base_distance_penalty: Penalidade base para cada rota proibida encontrada

    Retorna:
    - float: Distância total do caminho + penalidades por rotas proibidas
    """
    from genetic_algorithm import calculate_distance, calculate_fitness

    # Calcula a distância normal do caminho
    base_fitness = calculate_fitness(path)

    # Adiciona penalidades para rotas proibidas
    penalty = 0.0
    n = len(path)
    for i in range(n):
        city1 = path[i]
        city2 = path[(i + 1) % n]  # Conecta de volta à primeira cidade

        if forbidden_routes.is_route_forbidden(city1, city2):
            # Aplica uma penalidade proporcional à distância da rota proibida
            route_distance = calculate_distance(city1, city2)
            penalty += base_distance_penalty * route_distance

    return base_fitness + penalty
