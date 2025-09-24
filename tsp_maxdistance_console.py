import random
import itertools
from genetic_algorithm_max_distance import (
    mutate_mtsp, order_crossover_mtsp, generate_random_population_mtsp,
    calculate_fitness_mtsp, sort_population_mtsp, generate_nearest_neighbor_mtsp,
    N_AMBULANCES, MAX_DISTANCE_PER_AMBULANCE
)
from restrictions.max_distance_constraint import calculate_route_distance
import numpy as np
from benchmark_att48 import *

# GA para mTSP
N_CITIES = 48
POPULATION_SIZE = 100
N_GENERATIONS = 100
N_EXPLORATION_GENERATION = 50

# MUTATION
INITIAL_MUTATION_PROBABILITY = 0.85
AFTER_EXPLORATION_MUTATION_PROBABILITY = 0.5

# Initialize problem
# Using att48 benchmark
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)
scale_x = 1.0  # Sem escala para console
scale_y = 1.0
cities_locations = [(point[0], point[1]) for point in att_cities_locations]

print(f"Max distance per ambulance: {MAX_DISTANCE_PER_AMBULANCE}")
print(f"Number of ambulances: {N_AMBULANCES}")

generation_counter = itertools.count(start=1)
mutation_probability = INITIAL_MUTATION_PROBABILITY

# Create Initial Population para mTSP
population = generate_random_population_mtsp(cities_locations, POPULATION_SIZE - 10)

# Initialize with some nearest neighbor
for _ in range(10):
    nearest_routes = generate_nearest_neighbor_mtsp(cities_locations)
    population.append(nearest_routes)

best_fitness_values = []
best_solutions = []

# Main loop
for generation in range(1, N_GENERATIONS + 1):
    # Calculate fitness and sort population
    population_fitness = [calculate_fitness_mtsp(individual) for individual in population]
    population, population_fitness = sort_population_mtsp(population, population_fitness)

    best_fitness = population_fitness[0]
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    # Check route distances
    route_distances = [calculate_route_distance(route) for route in best_solution]
    print(f"Route distances: {[round(d, 2) for d in route_distances]}")
    violations = [d for d in route_distances if d > MAX_DISTANCE_PER_AMBULANCE]
    if violations:
        print(f"VIOLATIONS: {violations}")
    else:
        print("All routes within limit!")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    # Exploration time
    if generation > N_EXPLORATION_GENERATION:
        mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY

    while len(new_population) < POPULATION_SIZE:
        # Selection - simplificado para mTSP
        # Escolhe pais dos melhores 20%
        top_indices = sorted(range(len(population_fitness)), key=lambda i: population_fitness[i])[:len(population)//5]
        parent1 = population[random.choice(top_indices)]
        parent2 = population[random.choice(top_indices)]

        # Crossover
        child1, child2 = order_crossover_mtsp(parent1, parent2)

        # Mutation
        child1 = mutate_mtsp(child1, mutation_probability)
        child2 = mutate_mtsp(child2, mutation_probability)

        new_population.append(child1)
        new_population.append(child2)

    population = new_population

print("Final best fitness:", round(best_fitness, 2))
print("Final route distances:", [round(calculate_route_distance(route), 2) for route in best_solution])