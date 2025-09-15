

import random
import math
import copy 
from typing import List, Tuple
import numpy as np

default_problems = {
5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
10:[(470, 169), (602, 202), (754, 239), (476, 233), (468, 301), (522, 29), (597, 171), (487, 325), (746, 232), (558, 136)],
12:[(728, 67), (560, 160), (602, 312), (712, 148), (535, 340), (720, 354), (568, 300), (629, 260), (539, 46), (634, 343), (491, 135), (768, 161)],
15:[(512, 317), (741, 72), (552, 50), (772, 346), (637, 12), (589, 131), (732, 165), (605, 15), (730, 38), (576, 216), (589, 381), (711, 387), (563, 228), (494, 22), (787, 288)],
50:[(100, 50), (150, 80), (200, 120), (250, 90), (300, 150), (350, 110), (400, 180), (450, 140), (500, 200), (550, 160), (600, 220), (650, 190), (700, 250), (750, 210), (800, 280), (850, 240), (900, 310), (950, 270), (1000, 340), (1050, 300), (1100, 370), (1150, 330), (1200, 400), (1250, 360), (1300, 430), (1350, 390), (1400, 460), (1450, 420), (1500, 490), (1550, 450), (1600, 520), (1650, 480), (1700, 550), (1750, 510), (1800, 580), (1850, 540), (1900, 610), (1950, 570), (2000, 640), (2050, 600), (2100, 670), (2150, 630), (2200, 700), (2250, 660), (2300, 730), (2350, 690), (2400, 760), (2450, 720), (2500, 790), (2550, 750), (2600, 820), (2650, 780), (2700, 850), (2750, 810), (2800, 880), (2850, 840), (2900, 910), (2950, 870), (3000, 940)],
100:[(50, 50), (120, 80), (200, 120), (280, 90), (350, 150), (420, 110), (480, 180), (550, 140), (620, 200), (680, 160), (750, 220), (820, 190), (890, 250), (950, 210), (1020, 280), (1080, 240), (100, 300), (180, 340), (250, 380), (320, 320), (390, 360), (460, 400), (530, 440), (600, 480), (670, 520), (740, 560), (810, 600), (880, 640), (950, 680), (1020, 720), (1080, 760), (150, 800), (220, 840), (290, 880), (360, 920), (430, 960), (500, 1000), (570, 1040), (640, 1080), (710, 1020), (780, 980), (850, 940), (920, 900), (990, 860), (1060, 820), (80, 1000), (160, 960), (240, 920), (310, 880), (380, 840), (450, 800), (520, 760), (590, 720), (660, 680), (730, 640), (800, 600), (870, 560), (940, 520), (1010, 480), (1080, 440), (130, 400), (210, 360), (290, 320), (370, 280), (440, 240), (510, 200), (580, 160), (650, 120), (720, 80), (790, 40), (860, 100), (930, 140), (1000, 180), (1070, 220), (70, 600), (140, 640), (220, 680), (300, 720), (370, 760), (440, 800), (510, 840), (580, 880), (650, 920), (720, 960), (790, 1000), (860, 1040), (930, 1080), (1000, 1020), (1070, 980), (90, 200), (170, 240), (250, 280), (330, 320), (410, 360), (490, 400), (570, 440), (650, 480), (730, 520), (810, 560), (890, 600), (970, 640), (1050, 680), (60, 800), (140, 760), (220, 720), (300, 680), (380, 640), (460, 600), (540, 560), (620, 520), (700, 480), (780, 440), (860, 400), (940, 360), (1020, 320), (1080, 280)]
}

RESTRICTIONS_CONFIG={
    "COST_RESTRICTION": True
}

def generate_random_population(cities_location: List[Tuple[float, float]], population_size: int) -> List[List[Tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (List[Tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    List[List[Tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [random.sample(cities_location, len(cities_location)) for _ in range(population_size)]

def generate_nearest_neightbor(cities_location: List[Tuple[float, float]], initial_city: int = 0) -> List[Tuple[float, float]]:
    local_list =  copy.deepcopy(cities_location)

    initial_population = [local_list[initial_city]]
    local_list.pop(initial_city)
    
    while local_list:
        current_city = initial_population[-1]  # Last city in the path
        lowest_distance = float('inf')
        lowest_distance_city = None
        lowest_distance_index = -1

        for index, city in enumerate(local_list):            
            distance = calculate_distance(initial_population[len(initial_population) - 1], city)

            if distance < lowest_distance and not(city in initial_population):
                lowest_distance = distance
                lowest_distance_city = city
                lowest_distance_index = index

            if index == len(local_list) - 1:
                initial_population.append(lowest_distance_city)
                local_list.pop(lowest_distance_index)

    return initial_population

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (Tuple[float, float]): The coordinates of the first point.
    - point2 (Tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: List[Tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance.

    Parameters:
    - path (List[Tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total Euclidean distance of the path.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


def order_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (List[Tuple[float, float]]): The first parent sequence.
    - parent2 (List[Tuple[float, float]]): The second parent sequence.

    Returns:
    List[Tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child_parent1 = parent1[start_index:end_index]
    child_parent2 = parent2[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
    remaining_genes_parent2 = [gene for gene in parent2 if gene not in child_parent1]
    remaining_genes_parent1 = [gene for gene in parent1 if gene not in child_parent2]

    for position, gene in zip(remaining_positions, remaining_genes_parent2):
        child_parent1.insert(position, gene)

    for position, gene in zip(remaining_positions, remaining_genes_parent1):
        child_parent2.insert(position, gene)

    return child_parent1, child_parent2

### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]

def uniform_crossover(parent1: List[Tuple[float, float]], parent2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Uniform crossover for TSP that preserves all cities.
    """
    length = len(parent1)
    random_decision = [random.randint(0, 1) for _ in range(length)]
    
    child1 = []
    child2 = []
    
    # First pass: build children based on decisions
    for i in range(length):
        if random_decision[i] == 0:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    # Second pass: fix duplicates and missing cities
    child1 = fix_tsp_solution(child1, parent1, parent2)
    child2 = fix_tsp_solution(child2, parent1, parent2)
    
    return child1, child2

def fix_tsp_solution(child, parent1, parent2):
    """
    Fix a TSP solution by ensuring all cities are present exactly once.
    Uses both parents to provide alternative cities.
    """
    all_cities = set(parent1)  # All cities that should be present
    child_set = set(child)
    
    # Find missing cities
    missing_cities = list(all_cities - child_set)
    
    # Find duplicate cities and their positions
    city_positions = {}
    duplicates = []
    
    for i, city in enumerate(child):
        if city in city_positions:
            duplicates.append(i)  # Mark duplicate positions
        else:
            city_positions[city] = i
    
    # Create a pool of cities from both parents for replacement
    # Prioritize cities that appear in both parents
    both_parents_cities = set(parent1) & set(parent2)
    parent1_only_cities = set(parent1) - set(parent2)
    parent2_only_cities = set(parent2) - set(parent1)
    
    # Order missing cities by priority
    priority_missing = []
    for city in missing_cities:
        if city in both_parents_cities:
            priority_missing.append(city)
    
    for city in missing_cities:
        if city in parent1_only_cities or city in parent2_only_cities:
            priority_missing.append(city)
    
    # Replace duplicates with missing cities in priority order
    for i, dup_pos in enumerate(duplicates):
        if i < len(priority_missing):
            child[dup_pos] = priority_missing[i]
    
    return child


# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two. 
def mutate(solution:  List[Tuple[float, float]], mutation_probability: float) ->  List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur    
    if random.random() < mutation_probability:
        
        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution
    
        # Select a random index (excluding the last index) for swapping
        index = random.randint(0, len(solution) - 2)
        
        # Swap the cities at the selected index and the next index
        mutated_solution[index], mutated_solution[index + 1] = solution[index + 1], solution[index]   
        
    return mutated_solution

# mutate with intensity and suffle a part of the array instead of just swap
def mutate_hard(solution:  List[Tuple[float, float]], mutation_probability: float, intensity:int = 7) ->  List[Tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (List[int]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    List[int]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    MUTATE_QUANTITY = intensity

    # Check if mutation should occur    
    if random.random() < mutation_probability:
        
        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution
    
        # Select a random index (excluding the last index) for swapping
        start_index = random.randint(0, len(solution) - 2)
        pontentialy_final_index = start_index + MUTATE_QUANTITY
        end_index = pontentialy_final_index if pontentialy_final_index < len(mutated_solution) else len(mutated_solution) - 1

        subarray = mutated_solution[start_index:end_index]
        np.random.shuffle(subarray)
        mutated_solution[start_index:end_index] = subarray
        
        # Swap the cities at the selected index and the next index
        
    return mutated_solution

### Demonstration: mutation test code    
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(population: List[List[Tuple[float, float]]], fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (List[List[Tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (List[float]): The corresponding fitness values for each solution in the population.

    Returns:
    Tuple[List[List[Tuple[float, float]]], List[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return sorted_population, sorted_fitness


if __name__ == '__main__':
    N_CITIES = 10
    
    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [(random.randint(0, 100), random.randint(0, 100))
              for _ in range(N_CITIES)]
    
    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(N_GENERATIONS):
  
        
        population_fitness = [calculate_fitness(individual) for individual in population]    
        
        population, population_fitness = sort_population(population,  population_fitness)
        
        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]
           
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)    

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM
        
        while len(new_population) < POPULATION_SIZE:
            
            # SELECTION
            parent1, parent2 = random.choices(population[:10], k=2)  # Select parents from the top 10 individuals
            
            # CROSSOVER
            child1, child2 = order_crossover(parent1, parent2)
            
            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)
            
            new_population.append(child1)
            
    
        print('generation: ', generation)
        population = new_population
    


