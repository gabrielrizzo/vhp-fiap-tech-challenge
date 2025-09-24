# Funções auxiliares para mTSP
def inject_random_individuals_mtsp(population, cities_locations, injection_rate):
    """Injeta indivíduos aleatórios na população mTSP."""
    n_inject = int(len(population) * injection_rate)
    for _ in range(n_inject):
        population[random.randint(0, len(population)-1)] = generate_random_population_mtsp(cities_locations, 1)[0]
    return population

def inject_heuristic_individuals_mtsp(population, cities_locations, injection_rate):
    """Injeta indivíduos heurísticos na população mTSP."""
    n_inject = int(len(population) * injection_rate)
    for _ in range(n_inject):
        population[random.randint(0, len(population)-1)] = generate_nearest_neighbor_mtsp(cities_locations)
    return population

import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm_max_distance import (
    mutate_mtsp, order_crossover_mtsp, generate_random_population_mtsp,
    calculate_fitness_mtsp, sort_population_mtsp, generate_nearest_neighbor_mtsp,
    N_AMBULANCES, MAX_DISTANCE_PER_AMBULANCE
)
from draw_functions import draw_routes, draw_plot, draw_cities
from selection_functions import random_fitness_probability, tournament_selection, rank_based_selection
from helper_functions import (
    get_duplicated_items,
    lightweight_monitor_diversity,
    simple_diversity_aware_mutation,
    fast_diversity_aware_selection,
    population_edge_diversity,
    inject_random_individuals,
    inject_heuristic_individuals
)
import sys
import numpy as np
import pygame
from benchmark_att48 import *

# Define constant values
# pygame
WIDTH, HEIGHT = 1500, 800
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450
GENERATION_LIMIT = 10000

# GA para mTSP
N_CITIES = 48
POPULATION_SIZE = 1000
N_GENERATIONS = None
N_EXPLORATION_GENERATION = 1000

# NEAREST NEIGHBOR
N_NEIGHTBORS = 200

# MUTATION
INITIAL_MUTATION_PROBABILITY = 0.85
INTIAL_MUTATION_INTENSITY = 25
AFTER_EXPLORATION_MUTATION_INTENSITY = 5
AFTER_EXPLORATION_MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)

# Cores para as rotas das ambulâncias
ROUTE_COLORS = [BLUE, GREEN, YELLOW, PURPLE, ORANGE]

# Initialize problem
# Using att48 benchmark
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)
scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
scale_y = HEIGHT / max_y
cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
                     int(point[1] * scale_y)) for point in att_cities_locations]
target_solution = [cities_locations[i-1] for i in att_48_cities_order]
# Para mTSP, o target seria múltiplas rotas, mas simplificamos
print(f"Max distance per ambulance: {MAX_DISTANCE_PER_AMBULANCE}")
print(f"Number of ambulances: {N_AMBULANCES}")

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver with Max Distance Constraint (mTSP)")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)
generation_same_fitness_counter = 0
mutation_intensity = INTIAL_MUTATION_INTENSITY
mutation_probability = INITIAL_MUTATION_PROBABILITY

# Create Initial Population para mTSP
population = generate_random_population_mtsp(cities_locations, POPULATION_SIZE - N_NEIGHTBORS)

# Initialize with N_NEIGHTBORS usando nearest neighbor
for index in range(N_NEIGHTBORS):
    nearest_routes = generate_nearest_neighbor_mtsp(cities_locations)
    population.append(nearest_routes)

best_fitness_values = []
best_solutions = []
last_best_fitness = None

# Diversity management variables
diversity_history = []
diversity_threshold = 0.3
generation_without_improvement = 0
finished_exploration = False

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    generation = next(generation_counter)
    screen.fill(WHITE)

    # Calculate fitness and sort population
    population_fitness = [calculate_fitness_mtsp(individual) for individual in population]
    population, population_fitness = sort_population_mtsp(population, population_fitness)

    best_fitness = population_fitness[0]
    best_solution = population[0]

    if finished_exploration:
        if last_best_fitness is None or best_fitness < last_best_fitness:
            last_best_fitness = best_fitness
            generation_without_improvement = 0
        else:
            generation_without_improvement += 1

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    # Calculate current diversity (simplificado)
    current_diversity = 0.5  # Placeholder

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    # Simples injeção de diversidade se estagnado
    if generation_without_improvement > 200:
        print('Injecting diversity')
        population = inject_random_individuals_mtsp(population, cities_locations, 0.2)
        generation_without_improvement = 0

    # Monitor and manage diversity
    if current_diversity < diversity_threshold and generation % 50 == 0:
        if random.random() < 0.5:
            print('Injecting random individuals')
            # Para mTSP, injetar indivíduos aleatórios
            population = inject_random_individuals_mtsp(population, cities_locations, 0.1)
        else:
            print('Injecting heuristic individuals')
            population = inject_heuristic_individuals_mtsp(population, cities_locations, 0.1)

    new_population = [population[0]]  # Keep the best individual: ELITISM

    # Exploration time
    if generation > N_EXPLORATION_GENERATION and not finished_exploration:
        print('==== FINISHED EXPLORATION ===')
        mutation_intensity = AFTER_EXPLORATION_MUTATION_INTENSITY
        mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY
        finished_exploration = True

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

    # Adaptive parameter adjustment
    if generation_without_improvement > 100:
        if mutation_intensity < N_CITIES // 2:
            mutation_intensity += 1
            print(f"INCREASING MUTATION INTENSITY TO {mutation_intensity}")

        if mutation_probability < 0.9:
            mutation_probability += 0.02
            print(f"INCREASING MUTATION PROB TO {mutation_probability:.2f}")

        generation_without_improvement = 0

    # Draw visualization
    draw_plot(screen, list(range(len(best_fitness_values))), best_fitness_values, y_label="Fitness - Total Distance + Penalty")
    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    # Draw best routes with different colors
    colors = ROUTE_COLORS[:len(best_solution)]
    draw_routes(screen, best_solution, colors, width=3)
    # Draw second best for comparison
    if len(population) > 1:
        second_best = population[1]
        colors2 = [(c[0]//2, c[1]//2, c[2]//2) for c in ROUTE_COLORS[:len(second_best)]]  # Cores mais escuras
        draw_routes(screen, second_best, colors2, width=1)

    pygame.display.flip()
    clock.tick(FPS)

    if generation >= GENERATION_LIMIT:
        print('Generation limit reached. Stopping algorithm')
        running = False

# Exit software
pygame.quit()
sys.exit()