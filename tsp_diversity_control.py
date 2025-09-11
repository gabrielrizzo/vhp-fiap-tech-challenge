import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems, generate_nearest_neightbor, mutate_hard, uniform_crossover
from draw_functions import draw_paths, draw_plot, draw_cities
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
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450
GENERATION_LIMIT = 10000

# GA
N_CITIES = 48
POPULATION_SIZE = 2000  # Increased back to reasonable size
N_GENERATIONS = None
N_EXPLORATION_GENERATION = 1000

# NEAREST NEIGHTBOOR
N_NEIGHTBORS = 500  # Reduced to reasonable number

# MUTATION
INITIAL_MUTATION_PROBABILITY = 0.85
INTIAL_MUTATION_INTENSITY = 25  # Reduced for better performance
AFTER_EXPLORATION_MUTATION_INTENSITY = 5
AFTER_EXPLORATION_MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# Initialize problem
# Using att48 benchmark
WIDTH, HEIGHT = 1500, 800
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)
scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
scale_y = HEIGHT / max_y
cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
                     int(point[1] * scale_y)) for point in att_cities_locations]
target_solution = [cities_locations[i-1] for i in att_48_cities_order]
fitness_target_solution = calculate_fitness(target_solution)
print(f"Best Solution: {fitness_target_solution}")
print(f"Initial mutation intensity {INTIAL_MUTATION_INTENSITY} Initial mutation prob {INITIAL_MUTATION_PROBABILITY}")


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)
generation_same_fitness_counter = 0
mutation_intensity = INTIAL_MUTATION_INTENSITY
mutation_probability = INITIAL_MUTATION_PROBABILITY

# Create Initial Population
population = generate_random_population(cities_locations, POPULATION_SIZE - N_NEIGHTBORS)

# Initialize with N_NEIGHTBORS
for index, _ in enumerate([None for _ in range(N_NEIGHTBORS)]):
    nearest_neightbor = generate_nearest_neightbor(cities_locations, random.randint(0, N_CITIES - 1))
    population.append(nearest_neightbor)

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
    population_fitness = [calculate_fitness(individual) for individual in population]
    population, population_fitness = sort_population(population, population_fitness)

    best_fitness = calculate_fitness(population[0])
    best_solution = population[0]

    if finished_exploration:
        # Track improvements after exploration
        if last_best_fitness is None or best_fitness < last_best_fitness:
            last_best_fitness = best_fitness
            generation_without_improvement = 0
        else:
            generation_without_improvement += 1

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    # Draw visualization
    draw_plot(screen, list(range(len(best_fitness_values))), best_fitness_values, y_label="Fitness - Distance (pxls)")
    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    if len(population) > 1:
        draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    # Calculate current diversity
    current_diversity = population_edge_diversity(population)
    
    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}, Diversity = {current_diversity:.3f}")

    # Monitor and manage diversity (lightweight version)
    # population, mutation_probability, mutation_intensity, diversity_history = lightweight_monitor_diversity(
    #     population, generation, diversity_history, 
    #     mutation_probability, mutation_intensity
    # )

    # Inject diversity if needed
    if current_diversity < diversity_threshold and generation % 50 == 0:
        if random.random() < 0.5:
            print('inject_random_individuals')
            population = inject_random_individuals(population, cities_locations, 0.1)
        else:
            print('inject_heuristic_individuals')
            population = inject_heuristic_individuals(population, cities_locations, 0.1)

    new_population = [population[0]]  # Keep the best individual: ELITISM

    # Exploration time, running high mutation prob and intensity in the begining
    # We do it to make the algorithm explore not only optimal solutions, but also other solutions
    # that are not optimal but are still good or to include a good genetic data.
    if generation > N_EXPLORATION_GENERATION and not finished_exploration:
        print('==== FINISHED EXPLORATION ===')
        mutation_intensity = AFTER_EXPLORATION_MUTATION_INTENSITY
        mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY
        finished_exploration = True

    while len(new_population) < POPULATION_SIZE:
        # Use diversity-aware selection
        parent1, parent2 = fast_diversity_aware_selection(population, population_fitness, current_diversity)
        # parent1, parent2 = championship_selection(population)   
        # Crossover
        child1, child2 = order_crossover(parent1, parent2)

        # Adaptive mutation based on diversity
        if current_diversity < diversity_threshold:
            # Low diversity - use diversity-aware mutation
            print('=== LOW DIVERSITY DETECTED. DEPLOYING SOME BOMBS ====')
            child1 = simple_diversity_aware_mutation(child1, mutation_probability, mutation_intensity, current_diversity)
            child2 = simple_diversity_aware_mutation(child2, mutation_probability, mutation_intensity, current_diversity)
        else:
            # Normal diversity - use regular mutation
            child1 = mutate_hard(child1, mutation_probability, intensity=mutation_intensity)
            child2 = mutate_hard(child2, mutation_probability, intensity=mutation_intensity)

        new_population.append(child1)
        new_population.append(child2)

    population = new_population

    # Adaptive parameter adjustment Garantee that the algorithm will not get stuck in a local minimum
    if generation_without_improvement > 100:
        if mutation_intensity < N_CITIES // 2:
            mutation_intensity += 1
            print(f"INCREASING MUTATION INTENSITY TO {mutation_intensity}")
        
        if mutation_probability < 0.9:
            mutation_probability += 0.02
            print(f"INCREASING MUTATION PROB TO {mutation_probability:.2f}")

        generation_without_improvement = 0

    pygame.display.flip()
    clock.tick(FPS)

    if generation >= GENERATION_LIMIT:
        print('Generation limit reached. Stopping algorithm')
        running = False

# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
