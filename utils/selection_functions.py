import numpy as np
import random
from typing import List, Tuple

def calculate_fitness(individual):
    # This should be imported from genetic algorithm
    distance = 0
    n = len(individual)
    for i in range(n):
        current = individual[i]
        next_point = individual[(i + 1) % n]
        distance += ((current[0] - next_point[0]) ** 2 + (current[1] - next_point[1]) ** 2) ** 0.5
    return distance

def random_fitness_probability(population_fitness, population):
    probability = 1 / np.array(population_fitness)
    parent1, parent2 = random.choices(population, weights=probability, k=2)
    return parent1, parent2

def get_best_random_parent(population: List[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
    parent1, parent2 = random.choices(population, k=2)
    fitness_parent1 = calculate_fitness(parent1)
    fitness_parent2 = calculate_fitness(parent2)
    best_parent = parent1 if fitness_parent2 > fitness_parent1 else parent2
    return best_parent

def tournament_selection(population: List[Tuple[float, float]]) -> Tuple[Tuple[float, float]]:
    parent1 = get_best_random_parent(population)
    parent2 = get_best_random_parent(population)
    return parent1, parent2

def rank_based_selection(population: List[Tuple[float, float]], population_fitness: List[float]) -> Tuple[Tuple[float, float]]:
    sorted_indices = np.argsort(population_fitness)
    n = len(population)
    selection_probs = np.zeros(n)
    
    for rank, original_idx in enumerate(sorted_indices):
        selection_probs[original_idx] = 1.0 / (rank + 1)
    
    selection_probs = selection_probs / np.sum(selection_probs)
    
    parent1_idx = np.random.choice(len(population), p=selection_probs)
    parent2_idx = np.random.choice(len(population), p=selection_probs)
    
    return population[parent1_idx], population[parent2_idx]

def tournament_or_rank_based_selection(population: List[Tuple[float, float]], population_fitness: List[float], tournament_prob: float = 0.7) -> Tuple[Tuple[float, float]]:
    if random.random() < tournament_prob:  
        parent1, parent2 = tournament_selection(population)
    else:
        parent1, parent2 = rank_based_selection(population, population_fitness)
    return parent1, parent2
