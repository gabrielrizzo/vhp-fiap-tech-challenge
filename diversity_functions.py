import random
from selection_functions import calculate_fitness
from genetic_algorithm import generate_nearest_neightbor
from genetic_algorithm import mutate_hard

def inject_random_individuals(population, cities_locations, injection_rate=0.1):
    """Inject random individuals to increase diversity"""
    n_inject = int(len(population) * injection_rate)
    
    if n_inject == 0:
        return population
    
    # Generate random individuals
    random_individuals = []
    for _ in range(n_inject):
        random_individuals.append(random.sample(cities_locations, len(cities_locations)))
    
    # Replace worst individuals
    population.sort(key=lambda x: calculate_fitness(x))
    population[-n_inject:] = random_individuals
    
    print(f"Injected {n_inject} random individuals for diversity")
    return population

def inject_heuristic_individuals(population, cities_locations, injection_rate=0.1):
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
    population.sort(key=lambda x: calculate_fitness(x))
    population[-n_inject:] = heuristic_individuals

    print(f"Injected {n_inject} heuristic individuals for diversity")
    return population

def adaptive_mutation_intensification(population, base_mutation_prob, base_intensity, diversity_level):
    """Increase mutation when diversity is low"""
    if diversity_level < 0.3:  # Low diversity threshold
        # Increase mutation probability
        new_mutation_prob = min(1.0, base_mutation_prob * 1.5)
        
        # Increase mutation intensity
        new_intensity = min(len(population[0]), base_intensity * 2)
        
        print(f"Low diversity detected ({diversity_level:.3f})")
        print(f"Increasing mutation: prob {base_mutation_prob:.2f} -> {new_mutation_prob:.2f}")
        print(f"Increasing intensity: {base_intensity} -> {new_intensity}")
        
        return new_mutation_prob, new_intensity
    
    return base_mutation_prob, base_intensity

def diversity_aware_mutation(population, mutation_prob, intensity, diversity_level):
    """Apply different mutation strategies based on diversity"""
    mutated_population = []
    
    for individual in population:
        if random.random() < mutation_prob:
            if diversity_level < 0.2:  # Very low diversity
                # Use aggressive mutation
                mutated = aggressive_mutation(individual, intensity * 2)
            elif diversity_level < 0.4:  # Low diversity
                # Use moderate mutation
                mutated = moderate_mutation(individual, intensity)
            else:  # Good diversity
                # Use normal mutation
                mutated = mutate_hard(individual, 1.0, intensity)
        else:
            mutated = individual
        
        mutated_population.append(mutated)
    
    return mutated_population

def aggressive_mutation(individual, intensity):
    """Aggressive mutation for low diversity situations"""
    # Multiple mutations
    mutated = individual.copy()
    n_mutations = random.randint(2, 4)
    
    for _ in range(n_mutations):
        # Random segment reversal
        start = random.randint(0, len(mutated) - 2)
        end = random.randint(start + 1, len(mutated))
        mutated[start:end] = reversed(mutated[start:end])
        
        # Random swaps
        for _ in range(intensity // 2):
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
    
    return mutated

def moderate_mutation(individual, intensity):
    """Moderate mutation for low diversity situations"""
    return mutate_hard(individual, 1.0, intensity)