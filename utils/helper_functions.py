import numpy as np
import random
from typing import List, Tuple

def get_duplicated_items(list):
    duplicates = [i for i in set(list) if list.count(i) > 1]
    return duplicates

def fast_edge_diversity(solution1: List[Tuple[float, float]], solution2: List[Tuple[float, float]]) -> float:
    n = len(solution1)
    sample_size = min(10, n)
    sample_indices = random.sample(range(n), sample_size)
    
    shared_edges = 0
    for i in sample_indices:
        edge1 = tuple(sorted([solution1[i], solution1[(i + 1) % n]]))
        edge2 = tuple(sorted([solution2[i], solution2[(i + 1) % n]]))
        
        if edge1 == edge2:
            shared_edges += 1
    
    return 1.0 - (shared_edges / sample_size)

def population_edge_diversity(population: List[Tuple[float, float]]) -> float:
    n = len(population)
    sample_size = min(20, n * (n - 1) // 2)
    
    if sample_size == 0:
        return 0
    
    total_diversity = 0
    comparisons = 0
    
    for _ in range(sample_size):
        i, j = random.sample(range(n), 2)
        total_diversity += fast_edge_diversity(population[i], population[j])
        comparisons += 1
    
    return total_diversity / comparisons

def inject_random_individuals(population: List[Tuple[float, float]], cities_locations: List[Tuple[float, float]], injection_rate: float = 0.1) -> List[Tuple[float, float]]:
    n_inject = int(len(population) * injection_rate)
    
    if n_inject == 0:
        return population
    
    random_individuals = []
    for _ in range(n_inject):
        random_individuals.append(random.sample(cities_locations, len(cities_locations)))
    
    population = sorted(population, key=lambda x: calculate_fitness(x))
    population[-n_inject:] = random_individuals
    
    print(f"Injected {n_inject} random individuals for diversity")
    return population

def inject_heuristic_individuals(population: List[Tuple[float, float]], cities_locations: List[Tuple[float, float]], injection_rate: float = 0.1) -> List[Tuple[float, float]]:
    n_inject = int(len(population) * injection_rate)
    
    if n_inject == 0:
        return population
    
    heuristic_individuals = []
    for _ in range(n_inject):
        start_city = random.randint(0, len(cities_locations) - 1)
        nn_solution = generate_nearest_neighbor(cities_locations, start_city)
        heuristic_individuals.append(nn_solution)
    
    population = sorted(population, key=lambda x: calculate_fitness(x))
    population[-n_inject:] = heuristic_individuals
    
    print(f"Injected {n_inject} heuristic individuals for diversity")
    return population

def simple_diversity_aware_mutation(individual: List[Tuple[float, float]], mutation_prob: float, intensity: int, diversity_level: float) -> List[Tuple[float, float]]:
    if random.random() < mutation_prob:
        if diversity_level < 0.3:
            mutated = individual.copy()
            for _ in range(3):
                mutated = mutate_hard(mutated, 1.0, intensity)
            return mutated
        else:
            return mutate_hard(individual, 1.0, intensity)
    
    return individual

def fast_diversity_aware_selection(population: List[Tuple[float, float]], population_fitness: List[float], diversity_level: float) -> Tuple[Tuple[float, float]]:
    if diversity_level < 0.3:
        return random.choice(population), random.choice(population)
    else:
        from utils.selection_functions import tournament_or_rank_based_selection
        parent1, parent2 = tournament_or_rank_based_selection(
            population, population_fitness,
            tournament_prob=0.4
        )
    
    return parent1, parent2

def calculate_fitness(individual):
    distance = 0
    n = len(individual)
    for i in range(n):
        current = individual[i]
        next_point = individual[(i + 1) % n]
        distance += ((current[0] - next_point[0]) ** 2 + (current[1] - next_point[1]) ** 2) ** 0.5
    return distance

def generate_nearest_neighbor(cities_locations, initial_city):
    import copy
    local_list = copy.deepcopy(cities_locations)
    initial_population = [local_list[initial_city]]
    local_list.pop(initial_city)
    
    while local_list:
        current_city = initial_population[-1]
        lowest_distance = float('inf')
        lowest_distance_city = None
        lowest_distance_index = -1

        for index, city in enumerate(local_list):
            distance = calculate_distance(initial_population[-1], city)

            if distance < lowest_distance and city not in initial_population:
                lowest_distance = distance
                lowest_distance_city = city
                lowest_distance_index = index

            if index == len(local_list) - 1:
                initial_population.append(lowest_distance_city)
                local_list.pop(lowest_distance_index)

    return initial_population

def calculate_distance(point1, point2):
    import math
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def mutate_hard(solution, mutation_probability, intensity=7):
    import copy
    mutated_solution = copy.deepcopy(solution)

    if random.random() < mutation_probability:
        if len(solution) < 2:
            return solution

        start_index = random.randint(0, len(solution) - 2)
        potentially_final_index = start_index + intensity
        end_index = potentially_final_index if potentially_final_index < len(mutated_solution) else len(mutated_solution) - 1

        subarray = mutated_solution[start_index:end_index]
        np.random.shuffle(subarray)
        mutated_solution[start_index:end_index] = subarray

    return mutated_solution
