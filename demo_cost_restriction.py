import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import order_crossover, generate_random_population, sort_population, generate_nearest_neightbor, mutate_hard, default_problems
from draw_functions import draw_paths, draw_plot, draw_cities
from selection_functions import tournament_or_rank_based_selection
import sys
import numpy as np
import pygame
from benchmark_att48 import *
from restrictions.cost_restriction import RouteCostRestriction


# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450
GENERATION_LIMIT = 20

# GA
N_CITIES = 15
POPULATION_SIZE = 2000
N_GENERATIONS = None
N_EXPLORATION_GENERATION = 300

# NEAREST NEIGHTBOOR
N_NEIGHTBORS = 100

# MUTATION
INTIAL_MUTATION_INTENSITY = 40
INITIAL_MUTATION_PROBABILITY = 0.85
AFTER_EXPLORATION_MUTATION_INTENSITY = 30
AFTER_EXPLORATION_MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Cost
route_costs = {
    ((512, 317), (741, 72)): 7,
    ((605, 15), (637, 12)): 1
}


# Initialize problem
# Using Random cities generation
# cities_locations = [(random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS), random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS))
#                     for _ in range(N_CITIES)]


# # Using Deault Problems: 10, 12 or 15
# WIDTH, HEIGHT = 800, 400
cities_locations = default_problems[15]


# Using att48 benchmark
# WIDTH, HEIGHT = 1500, 800
# att_cities_locations = np.array(att_48_cities_locations)
# max_x = max(point[0] for point in att_cities_locations)
# max_y = max(point[1] for point in att_cities_locations)
# scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
# scale_y = HEIGHT / max_y
# cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
#                      int(point[1] * scale_y)) for point in att_cities_locations]
# target_solution = [cities_locations[i-1] for i in att_48_cities_order]
print(f"Initial mutation intensity {INTIAL_MUTATION_INTENSITY} Initial mutation prob {INITIAL_MUTATION_PROBABILITY}")
# ----- Using att48 benchmark


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)  # Start the counter at 1
generation_same_fitness_counter = 0
mutation_intensity = INTIAL_MUTATION_INTENSITY
mutation_probability = INITIAL_MUTATION_PROBABILITY
finished_exploration = False

route_restriction = RouteCostRestriction(cities_locations, route_costs, gas_cost_per_km=5)
route_restriction.config_dimensions(
    width=1500, plot_x_offset=450, height=800, node_radius=10
)

# fitness_target_solution = route_restriction.united_fitness(
#     path=target_solution,
#     use_normalized=True)
# print(f"Best Solution: {fitness_target_solution}")

# Create Initial Population
# TODO:- use some heuristic like Nearest Neighbour our Convex Hull to initialize
population = generate_random_population(cities_locations, POPULATION_SIZE - N_NEIGHTBORS)

# Initialize with N_NEIGHTBORS
for index, _ in enumerate([None for _ in range(N_NEIGHTBORS)]):
    nearest_neightbor = generate_nearest_neightbor(cities_locations, random.randint(0,N_CITIES - 1))
    population.append(nearest_neightbor)

best_fitness_values = []
best_solutions = []
last_best_fitness = None


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

    population_fitness = [route_restriction.united_fitness(
    path=individual,
    use_normalized=False
) for individual in population]

    population, population_fitness = sort_population(
        population,  population_fitness)

    best_fitness = route_restriction.united_fitness(
    path=population[0],
    use_normalized=False)

    best_solution = population[0]

    if finished_exploration:
        if last_best_fitness == best_fitness:
            generation_same_fitness_counter = generation_same_fitness_counter + 1
        else:
            generation_same_fitness_counter = 0

    last_best_fitness = best_fitness
    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))),
              best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[20], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    # Exploration time, running high mutation prob and intensity in the begining
    if generation > N_EXPLORATION_GENERATION and not finished_exploration:
        print('==== FINISHED EXPLORATION ===')
        mutation_intensity = AFTER_EXPLORATION_MUTATION_INTENSITY
        mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY
        finished_exploration = True

    while len(new_population) < POPULATION_SIZE:

        # selection
        # simple selection based on first 10 best solutions
        # parent1, parent2 = random.choices(population[:10], k=2)

        # parent1, parent2 = random_fitness_probability(population_fitness=population_fitness, population=population)
        # parent1, parent2 = rank_based_selection(population, population_fitness=population_fitness)

         # Enhanced selection
        parent1, parent2 = tournament_or_rank_based_selection(
            population, population_fitness,
            tournament_prob=0.7
        )

        child1, child2 = order_crossover(parent1, parent2)
        # child1, child2 = uniform_crossover(parent1, parent2)

        if generation_same_fitness_counter == 100:
            if mutation_intensity < N_CITIES:
                mutation_intensity = mutation_intensity + 2
                # Reset the counter to let the mutation run for a new set of generation limit
                print(f"INCREASING MUTATION INTENSITY TO {mutation_intensity}")
        
            if mutation_probability < 0.85:
                mutation_probability = mutation_probability + 0.05
                # Reset the counter to let the mutation run for a new set of generation limit
                print(f"INCREASING MUTATION PROB TO {mutation_probability}")

            generation_same_fitness_counter = 0


        child1 = mutate_hard(child1, mutation_probability, intensity=mutation_intensity)

        child2 = mutate_hard(child2, mutation_probability, intensity=mutation_intensity)

        new_population.append(child1)

        new_population.append(child2)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)

    if generation >= GENERATION_LIMIT:
        print('Generation limit reached. Stopping algorithm')
        print(route_restriction.get_route_description(population[0], False))
        running = False


# TODO: save the best individual in a file if it is better than the one saved.

# exit software
pygame.quit()
sys.exit()
