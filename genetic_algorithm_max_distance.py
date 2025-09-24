import random
import math
import copy
from typing import List, Tuple
from scipy.spatial import ConvexHull
import numpy as np
from restrictions.max_distance_constraint import apply_max_distance_constraint, calculate_route_distance

# Constantes para mTSP
N_AMBULANCES = 3  # Número de ambulâncias
MAX_DISTANCE_PER_AMBULANCE = 250.0  # Distância máxima por ambulância

default_problems = {
    5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
    10: [(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
    12: [(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
    15: [(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)],
    50: [(100, 50), (150, 80), (200, 120), (250, 90), (300, 150), (350, 110), (400, 180), (450, 140), (500, 200), (550, 160), (600, 220), (650, 190), (700, 250), (750, 210), (800, 280), (850, 240), (900, 310), (950, 270), (1000, 340), (1050, 300), (1100, 370), (1150, 330), (1200, 400), (1250, 360), (1300, 430), (1350, 390), (1400, 460), (1450, 420), (1500, 490), (1550, 450), (1600, 520), (1650, 480), (1700, 550), (1750, 510), (1800, 580), (1850, 540), (1900, 610), (1950, 570), (2000, 640), (2050, 600), (2100, 670), (2150, 630), (2200, 700), (2250, 660), (2300, 730), (2350, 690), (2400, 760), (2450, 720), (2500, 790), (2550, 750), (2600, 820), (2650, 780), (2700, 850), (2750, 810), (2800, 880), (2850, 840), (2900, 910), (2950, 870), (3000, 940)],
    100: [(50, 50), (120, 80), (200, 120), (280, 90), (350, 150), (420, 110), (480, 180), (550, 140), (620, 200), (680, 160), (750, 220), (820, 190), (890, 250), (950, 210), (1020, 280), (1080, 240), (100, 300), (180, 340), (250, 380), (320, 320), (390, 360), (460, 400), (530, 440), (600, 480), (670, 520), (740, 560), (810, 600), (880, 640), (950, 680), (1020, 720), (1080, 760), (150, 800), (220, 840), (290, 880), (360, 920), (430, 960), (500, 1000), (570, 1040), (640, 1080), (710, 1020), (780, 980), (850, 940), (920, 900), (990, 860), (1060, 820), (80, 1000), (160, 960), (240, 920), (310, 880), (380, 840), (450, 800), (520, 760), (590, 720), (660, 680), (730, 640), (800, 600), (870, 560), (940, 520), (1010, 480), (1080, 440), (130, 400), (210, 360), (290, 320), (370, 280), (440, 240), (510, 200), (580, 160), (650, 120), (720, 80), (790, 40), (860, 100), (930, 140), (1000, 180), (1070, 220), (70, 600), (140, 640), (220, 680), (300, 720), (370, 760), (440, 800), (510, 840), (580, 880), (650, 920), (720, 960), (790, 1000), (860, 1040), (930, 1080), (1000, 1020), (1070, 980), (90, 200), (170, 240), (250, 280), (330, 320), (410, 360), (490, 400), (570, 440), (650, 480), (730, 520), (810, 560), (890, 600), (970, 640), (1050, 680), (60, 800), (140, 760), (220, 720), (300, 680), (380, 640), (460, 600), (540, 560), (620, 520), (700, 480), (780, 440), (860, 400), (940, 360), (1020, 320), (1080, 280)]
}

def generate_random_population_mtsp(cities_location: List[Tuple[float, float]], population_size: int, n_ambulances: int = N_AMBULANCES) -> List[List[List[Tuple[float, float]]]]:
    """
    Gera uma população aleatória de rotas para múltiplas ambulâncias (mTSP).

    Cada indivíduo é uma lista de rotas, uma para cada ambulância.
    Cada rota começa e termina na base (primeira cidade).

    Parameters:
    - cities_location: Lista de localizações das cidades
    - population_size: Tamanho da população
    - n_ambulances: Número de ambulâncias

    Returns:
    Lista de indivíduos, cada um sendo uma lista de rotas
    """
    base = cities_location[0]  # Assume base é a primeira cidade
    cities_to_assign = cities_location[1:]  # Exclui a base

    population = []
    for _ in range(population_size):
        # Embaralha as cidades
        shuffled_cities = random.sample(cities_to_assign, len(cities_to_assign))

        # Divide em grupos para cada ambulância
        routes = []
        cities_per_ambulance = len(shuffled_cities) // n_ambulances
        remainder = len(shuffled_cities) % n_ambulances

        start = 0
        for i in range(n_ambulances):
            end = start + cities_per_ambulance + (1 if i < remainder else 0)
            route_cities = shuffled_cities[start:end]
            # Adiciona base no início e fim
            route = [base] + route_cities + [base]
            routes.append(route)
            start = end

        population.append(routes)

    return population

def generate_nearest_neighbor_mtsp(cities_location: List[Tuple[float, float]], n_ambulances: int = N_AMBULANCES) -> List[List[Tuple[float, float]]]:
    """
    Gera uma solução inicial usando nearest neighbor para múltiplas ambulâncias.
    """
    base = cities_location[0]
    cities_to_assign = cities_location[1:]

    routes = []
    cities_copy = copy.deepcopy(cities_to_assign)

    for _ in range(n_ambulances):
        if not cities_copy:
            break

        route = [base]
        current_city = base

        while cities_copy:
            # Encontra a cidade mais próxima
            nearest_city = min(cities_copy, key=lambda c: calculate_distance(current_city, c))
            route.append(nearest_city)
            cities_copy.remove(nearest_city)
            current_city = nearest_city

            # Verifica se adicionar a próxima cidade excederia o limite
            if calculate_route_distance(route + [base]) > MAX_DISTANCE_PER_AMBULANCE:
                break

        route.append(base)
        routes.append(route)

    # Se sobrar cidades, distribui para as rotas existentes (simplificado)
    while cities_copy:
        for route in routes:
            if cities_copy:
                # Insere a cidade mais próxima na rota
                city = cities_copy.pop(0)
                # Simples: adiciona antes do último base
                route.insert(-1, city)

    return routes

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def calculate_fitness_mtsp(routes: List[List[Tuple[float, float]]]) -> float:
    """
    Calcula o fitness para mTSP, incluindo penalização por exceder distância máxima.
    """
    total_distance = sum(calculate_route_distance(route) for route in routes)
    penalty = apply_max_distance_constraint(routes, MAX_DISTANCE_PER_AMBULANCE)
    return total_distance + penalty

def order_crossover_mtsp(parent1: List[List[Tuple[float, float]]], parent2: List[List[Tuple[float, float]]]) -> Tuple[List[List[Tuple[float, float]]], List[List[Tuple[float, float]]]]:
    """
    Crossover para mTSP: aplica crossover em cada rota correspondente.
    """
    child1 = []
    child2 = []

    for r1, r2 in zip(parent1, parent2):
        c1, c2 = order_crossover(r1, r2)
        child1.append(c1)
        child2.append(c2)

    return child1, child2

def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Order crossover adaptado para rotas (sem base duplicada no meio).
    """
    length = len(parent1)
    if length < 3:
        return parent1.copy(), parent2.copy()

    # Escolhe pontos de crossover, evitando base
    start = random.randint(1, length - 2)
    end = random.randint(start + 1, length - 1)

    child1 = parent1[start:end]
    child2 = parent2[start:end]

    # Preenche o resto
    remaining1 = [c for c in parent2 if c not in child1]
    remaining2 = [c for c in parent1 if c not in child2]

    child1 = remaining1[:start] + child1 + remaining1[start:]
    child2 = remaining2[:start] + child2 + remaining2[start:]

    return child1, child2

def mutate_mtsp(routes: List[List[Tuple[float, float]]], mutation_probability: float) -> List[List[Tuple[float, float]]]:
    """
    Aplica mutação em cada rota.
    """
    return [mutate(route, mutation_probability) for route in routes]

def mutate(solution: List[Tuple[float, float]], mutation_probability: float) -> List[Tuple[float, float]]:
    """
    Mutação por troca de duas cidades (exceto base).
    """
    if random.random() < mutation_probability and len(solution) > 3:
        # Escolhe índices entre 1 e len-2 (evita base)
        i = random.randint(1, len(solution) - 2)
        j = random.randint(1, len(solution) - 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution

def sort_population_mtsp(population: List[List[List[Tuple[float, float]]]], fitness: List[float]) -> Tuple[List[List[List[Tuple[float, float]]]], List[float]]:
    combined = list(zip(population, fitness))
    combined.sort(key=lambda x: x[1])
    sorted_pop, sorted_fit = zip(*combined)
    return list(sorted_pop), list(sorted_fit)

if __name__ == '__main__':
    # Teste básico
    cities = default_problems[10]
    pop = generate_random_population_mtsp(cities, 5)
    fitnesses = [calculate_fitness_mtsp(ind) for ind in pop]
    print("Fitnesses:", fitnesses)