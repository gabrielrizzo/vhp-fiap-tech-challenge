import numpy as np
import random
from typing import List, Tuple
from genetic_algorithm import calculate_fitness
def random_fitness_probability(population_fitness, population):
    # solution based on fitness probability
    probability = 1 / np.array(population_fitness)
    parent1, parent2 = random.choices(population, weights=probability, k=2)

    return parent1, parent2

def get_best_random_parent(population: List[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
    parent1, parent2 = random.choices(population, k=2)

    fitness_parent1 = calculate_fitness(parent1)
    fitness_parent2 = calculate_fitness(parent2)

    best_parent = parent1 if fitness_parent2 > fitness_parent1 else parent2

    return best_parent

def championship_selection(population: List[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
    parent1 = get_best_random_parent(population)
    parent2 = get_best_random_parent(population)

    return parent1, parent2

def rank_based_selection(population: List[Tuple[float, float]], population_fitness: List[float]) -> Tuple[Tuple[float, float]]:
    """Rank-based selection for better diversity"""
    # Sort by fitness (ascending for TSP - lower distance is better)
    sorted_indices = np.argsort(population_fitness)
    
    # Create selection probabilities directly from sorted indices
    n = len(population)
    selection_probs = np.zeros(n)
    
    # Assign probabilities: best gets highest, worst gets lowest
    for rank, original_idx in enumerate(sorted_indices):
        selection_probs[original_idx] = 1.0 / (rank + 1)  # rank + 1 because rank starts at 0
    
    # Normalize probabilities
    selection_probs = selection_probs / np.sum(selection_probs)
    
    # Select parents
    parent1_idx = np.random.choice(len(population), p=selection_probs)
    parent2_idx = np.random.choice(len(population), p=selection_probs)
    
    return population[parent1_idx], population[parent2_idx]