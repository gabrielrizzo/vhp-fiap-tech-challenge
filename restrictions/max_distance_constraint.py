"""
Restrição: Distância Máxima Total por Ambulância

Propósito:
Garante que nenhuma ambulância percorra mais de 250 km em um único plantão,
forçando a divisão dos atendimentos entre múltiplas ambulâncias quando necessário.

Parâmetros:
- max_distance: Distância máxima permitida por ambulância (padrão: 250 km)
- penalty_factor: Fator de penalização para rotas que excedem o limite (padrão: 1000)

Como é aplicada:
- Calcula a distância de cada rota individual
- Aplica penalização se qualquer rota exceder o limite
- Afeta o fitness total somando penalizações

Impacto na rota final:
- Força a divisão de rotas longas entre múltiplas ambulâncias
- Pode aumentar o custo total devido à penalização, mas garante viabilidade prática
"""

from typing import List, Tuple
from genetic_algorithm import calculate_distance


def apply_max_distance_constraint(routes: List[List[Tuple[float, float]]],
                                max_distance: float = 250.0,
                                penalty_factor: float = 1000.0) -> float:
    """
    Aplica a restrição de distância máxima às rotas das ambulâncias.

    Parameters:
    - routes: Lista de rotas, onde cada rota é uma lista de cidades (Tuple[float, float])
    - max_distance: Distância máxima permitida por ambulância
    - penalty_factor: Fator multiplicador para penalização

    Returns:
    - Penalização total a ser adicionada ao fitness
    """
    total_penalty = 0.0

    for route in routes:
        route_distance = calculate_route_distance(route)
        if route_distance > max_distance:
            excess = route_distance - max_distance
            total_penalty += excess * penalty_factor

    return total_penalty


def calculate_route_distance(route: List[Tuple[float, float]]) -> float:
    """
    Calcula a distância total de uma rota (caminho aberto).

    Parameters:
    - route: Lista de cidades da rota

    Returns:
    - Distância total da rota
    """
    if len(route) < 2:
        return 0.0

    distance = 0.0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i + 1])

    return distance


def is_route_valid(route: List[Tuple[float, float]], max_distance: float = 250.0) -> bool:
    """
    Verifica se uma rota está dentro do limite de distância.

    Parameters:
    - route: Rota a verificar
    - max_distance: Limite de distância

    Returns:
    - True se válida, False caso contrário
    """
    return calculate_route_distance(route) <= max_distance


def split_route_by_distance(full_route: List[Tuple[float, float]],
                           max_distance: float = 250.0) -> List[List[Tuple[float, float]]]:
    """
    Divide uma rota completa em múltiplas rotas respeitando o limite de distância.
    Assume que a primeira cidade é a base e todas as rotas devem começar e terminar nela.

    Parameters:
    - full_route: Rota completa a ser dividida
    - max_distance: Distância máxima por segmento

    Returns:
    - Lista de rotas divididas
    """
    if not full_route:
        return []

    base = full_route[0]
    routes = []
    current_route = [base]
    current_distance = 0.0

    for i in range(1, len(full_route)):
        next_distance = calculate_distance(full_route[i-1], full_route[i])

        if current_distance + next_distance <= max_distance:
            current_route.append(full_route[i])
            current_distance += next_distance
        else:
            # Fecha a rota atual na base
            current_route.append(base)
            routes.append(current_route)

            # Inicia nova rota
            current_route = [base, full_route[i]]
            current_distance = calculate_distance(base, full_route[i])

    # Fecha a última rota
    if len(current_route) > 1:
        current_route.append(base)
        routes.append(current_route)

    return routes