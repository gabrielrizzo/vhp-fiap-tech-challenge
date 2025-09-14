
"""
Módulo de Restrições de Custo para o Problema TSP (Traveling Salesman Problem)

Este módulo implementa funcionalidades para calcular custos adicionais (como pedágios)
associados à visita de cidades específicas no problema TSP. O custo é adicionado ao
fitness original da solução, penalizando rotas que passam por cidades com custos elevados.

Funcionalidades:
- Mapeamento de custos para o benchmark ATT48
- Normalização de coordenadas para compatibilidade com o sistema de visualização
- Cálculo de fitness com custos adicionais
- Suporte para problemas customizados além do ATT48

Versão: 1.0
"""
from typing import List, Tuple, Dict
from benchmark_att48 import att_48_cities_locations

PLOT_X_OFFSET = 450
WIDTH, HEIGHT = 1500, 800
NODE_RADIUS = 10
# Cost restriction dict for ATT48 benchmark
BENCHMARK_ATT48_DICT = {
(6734, 1453): 100,
(2233 , 10): 100,
(5530, 1424): 100,
(401, 841): 100,
(3082, 1644): 100,
(7608, 4458): 100,
(7573, 3716): 100,
(7265, 1268): 100,
(6898, 1885): 50,
(1112, 2049): 50,
(5468, 2606): 200,
(5989, 2873): 0,
(4706, 2674): 0,
(4612, 2035): 0,
(6347, 2683): 0,
(6107, 669): 400,
(7611, 5184): 40,
(7462, 3590): 40,
(7732, 4723): 150,
(5900, 3561): 150,
(4483, 3369): 200,
(6101, 1110): 0,
(5199, 2182): 0,
(1633, 2809): 0,
(4307, 2322): 0,
 (675, 1006): 0,
(7555, 4819): 0,
(7541, 3981): 0,
(3177, 756): 0,
(7352, 4506): 0,
(7545, 2801): 0,
(3245, 3305): 0,
(6426, 3173): 0,
(4608, 1198): 0,
(23, 2216): 0,
(7248, 3779): 10,
(7762, 4595): 20,
(7392, 2244): 0,
(3484, 2829): 0,
(6271, 2135): 50,
(4985, 140): 0,
(1916, 1569): 35,
(7280, 4899): 0,
(7509, 3239): 0,
(10, 2676): 0,
(6807, 2993): 0,
(5185, 3258): 0,
(3023, 1942): 5
}

def create_normalized_cost_mapping():
    """
    Create a mapping from normalized coordinates to costs.
    This avoids the need to reverse normalization.
    @TODO: Create config file to don't repeat. Refactor inside this code and on tsp.py
    """
    # Calculate scaling factors (same as in tsp.py)
    max_x = max(point[0] for point in att_48_cities_locations)
    max_y = max(point[1] for point in att_48_cities_locations)
    scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
    scale_y = HEIGHT / max_y

    # Create mapping from normalized coordinates to costs
    normalized_cost_dict = {}
    for original_point, cost in BENCHMARK_ATT48_DICT.items():
        # Normalize the original point (same as in tsp.py)
        normalized_x = int(original_point[0] * scale_x + PLOT_X_OFFSET)
        normalized_y = int(original_point[1] * scale_y)
        normalized_point = (normalized_x, normalized_y)
        normalized_cost_dict[normalized_point] = cost

    return normalized_cost_dict

# Create the mapping once
NORMALIZED_COST_DICT = create_normalized_cost_mapping()

def calculate_fitness_with_cost(
    path: List[Tuple[float, float]],
    original_fitness: float,
    cost_city_mapping: Dict,
    is_att48: bool = True,
    ) -> float:
    """
	Calcula o fitness baseado no custo de chegar em cada cidade. Esse custo seria o valor gasto de
    pedágio para chegar em cada ponto.
    
    path: Lista de cidades da solução
    original_fitness: Valor original calculado do fitness
    cost_city_mapping: Opcional caso não seja o problema do att48. 
    Contem o dicionário com o custo de se chegar em cada cidade.
    is_att48: Se for o problema de att48 fazemos o calculo para normalizar as coordenadas.
    """

    total_cost = 0

    for city in path:
        if is_att48:
            total_cost = total_cost + NORMALIZED_COST_DICT.get(city, 0)
        else:
            total_cost = total_cost + cost_city_mapping.get(city, 0)

    return total_cost + original_fitness
