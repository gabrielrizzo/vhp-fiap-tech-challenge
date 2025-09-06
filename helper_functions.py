import numpy as np
import random
from typing import List, Tuple
from genetic_algorithm import generate_nearest_neightbor, calculate_fitness
from genetic_algorithm import mutate_hard
from selection_functions import tournament_or_rank_based_selection

def get_duplicated_items(list):
    duplicates = [i for i in set(list) if list.count(i) > 1]
    return duplicates

# Replace the existing functions with these lighter versions:

def fast_edge_diversity(solution1: List[Tuple[float, float]], solution2: List[Tuple[float, float]]) -> float:
    """Lightweight edge diversity calculation - only check a sample of edges"""
    n = len(solution1)
    sample_size = min(10, n)  # Only check 10 edges or all if less than 10
    
    # Sample random edges instead of checking all
    sample_indices = random.sample(range(n), sample_size)
    
    shared_edges = 0
    for i in sample_indices:
        # Check edge from i to i+1
        edge1 = tuple(sorted([solution1[i], solution1[(i + 1) % n]]))
        edge2 = tuple(sorted([solution2[i], solution2[(i + 1) % n]]))
        
        if edge1 == edge2:
            shared_edges += 1
    
    # Return diversity as percentage of different edges
    return 1.0 - (shared_edges / sample_size)

def population_edge_diversity(population: List[Tuple[float, float]]) -> float:
    """Lightweight population diversity - only check a sample of pairs"""
    n = len(population)
    
    # Only check a sample of pairs instead of all combinations
    sample_size = min(20, n * (n - 1) // 2)  # Max 20 comparisons
    
    if sample_size == 0:
        return 0
    
    total_diversity = 0
    comparisons = 0
    
    # Sample random pairs
    for _ in range(sample_size):
        i, j = random.sample(range(n), 2)
        total_diversity += fast_edge_diversity(population[i], population[j])
        comparisons += 1
    
    return total_diversity / comparisons

def inject_random_individuals(population: List[Tuple[float, float]], cities_locations: List[Tuple[float, float]], injection_rate: float = 0.1) -> List[Tuple[float, float]]:
    """Inject random individuals to increase diversity"""
    n_inject = int(len(population) * injection_rate)
    
    if n_inject == 0:
        return population
    
    # Generate random individuals
    random_individuals = []
    for _ in range(n_inject):
        random_individuals.append(random.sample(cities_locations, len(cities_locations)))
    
    # Replace worst individuals
    population = sorted(population, key=lambda x: calculate_fitness(x))
    population[-n_inject:] = random_individuals
    
    print(f"Injected {n_inject} random individuals for diversity")
    return population

def inject_heuristic_individuals(population: List[Tuple[float, float]], cities_locations: List[Tuple[float, float]], injection_rate: float = 0.1) -> List[Tuple[float, float]]:
    """Inject heuristic-based individuals (nearest neighbor, etc.)"""
    n_inject = int(len(population) * injection_rate)
    
    if n_inject == 0:
        return population
    
    # Generate nearest neighbor solutions from different starting points
    heuristic_individuals = []
    for _ in range(n_inject):
        start_city = random.randint(0, len(cities_locations) - 1)
        nn_solution = generate_nearest_neightbor(cities_locations, start_city)
        heuristic_individuals.append(nn_solution)
    
    # Replace worst individuals
    population = sorted(population, key=lambda x: calculate_fitness(x))
    population[-n_inject:] = heuristic_individuals
    
    print(f"Injected {n_inject} heuristic individuals for diversity")
    return population

def simple_diversity_aware_mutation(individual: List[Tuple[float, float]], mutation_prob: float, intensity: int, diversity_level: float) -> List[Tuple[float, float]]:
    """Simplified diversity-aware mutation - much faster"""
    if random.random() < mutation_prob:
        if diversity_level < 0.3:  # Low diversity - more aggressive
            # Just do multiple normal mutations
            mutated = individual.copy()
            for _ in range(3):  # 3 mutations instead of complex aggressive mutation
                mutated = mutate_hard(mutated, 1.0, intensity)
            return mutated
        else:  # Normal diversity - regular mutation
            return mutate_hard(individual, 1.0, intensity)
    
    return individual

def fast_diversity_aware_selection(population: List[Tuple[float, float]], population_fitness: List[float], diversity_level: float) -> Tuple[Tuple[float, float]]:
    parent1 = None
    parent2 = None
    """Simplified diversity-aware selection"""
    if diversity_level < 0.3:  # Low diversity - use random selection
        return random.choice(population), random.choice(population)
    else:  # Normal diversity - use the random logic to validate which selection will choose
        parent1, parent2 = tournament_or_rank_based_selection(
            population, population_fitness,
            tournament_prob=0.4
        )
    
    return parent1, parent2

def lightweight_monitor_diversity(population: List[Tuple[float, float]], generation: int, diversity_history: List[float], mutation_prob: float, mutation_intensity: int) -> Tuple[List[Tuple[float, float]], float, int, List[float]]:
    """Lightweight diversity monitoring - only check every 10 generations"""
    if generation % 10 != 0:  # Only check every 10 generations
        return population, mutation_prob, mutation_intensity, diversity_history
    
    current_diversity = population_edge_diversity(population)
    diversity_history.append(current_diversity)
    
    # Simple action based on diversity
    if current_diversity < 0.3:  # Low diversity
        print(f"Generation {generation}: Low diversity ({current_diversity:.3f}), increasing mutation")
        mutation_prob = min(1.0, mutation_prob * 1.2)
        mutation_intensity = min(len(population[0]), int(mutation_intensity * 1.3))
    
    return population, mutation_prob, mutation_intensity, diversity_history