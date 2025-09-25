import random
import math
import copy 
from typing import List, Tuple, Dict, Any
from scipy.spatial import ConvexHull
import numpy as np
from core.restriction_manager import RestrictionManager

class EnhancedGeneticAlgorithm:
    def __init__(self, cities_locations: List[Tuple[float, float]]):
        self.cities_locations = cities_locations
        self.restriction_manager = RestrictionManager()
        
    def calculate_base_fitness(self, path: List[Tuple[float, float]]) -> float:
        distance = 0
        n = len(path)
        for i in range(n):
            distance += self.calculate_distance(path[i], path[(i + 1) % n])
        return distance
    
    def calculate_fitness_with_restrictions(self, path: List[Tuple[float, float]], 
                                         vehicle_data: Dict[str, Any] = None) -> float:
        base_fitness = self.calculate_base_fitness(path)
        return self.restriction_manager.calculate_fitness_with_restrictions(
            path, base_fitness, vehicle_data
        )
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def generate_random_population(self, population_size: int) -> List[List[Tuple[float, float]]]:
        return [random.sample(self.cities_locations, len(self.cities_locations)) 
                for _ in range(population_size)]
    
    def generate_nearest_neighbor(self, initial_city: int = 0) -> List[Tuple[float, float]]:
        local_list = copy.deepcopy(self.cities_locations)
        initial_population = [local_list[initial_city]]
        local_list.pop(initial_city)
        
        while local_list:
            current_city = initial_population[-1]
            lowest_distance = float('inf')
            lowest_distance_city = None
            lowest_distance_index = -1

            for index, city in enumerate(local_list):            
                distance = self.calculate_distance(initial_population[-1], city)

                if distance < lowest_distance and city not in initial_population:
                    lowest_distance = distance
                    lowest_distance_city = city
                    lowest_distance_index = index

                if index == len(local_list) - 1:
                    initial_population.append(lowest_distance_city)
                    local_list.pop(lowest_distance_index)

        return initial_population
    
    def order_crossover(self, parent1: List[Tuple[float, float]], 
                       parent2: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        length = len(parent1)
        start_index = random.randint(0, length - 1)
        end_index = random.randint(start_index + 1, length)

        child_parent1 = parent1[start_index:end_index]
        child_parent2 = parent2[start_index:end_index]

        remaining_positions = [i for i in range(length) if i < start_index or i >= end_index]
        remaining_genes_parent2 = [gene for gene in parent2 if gene not in child_parent1]
        remaining_genes_parent1 = [gene for gene in parent1 if gene not in child_parent2]

        for position, gene in zip(remaining_positions, remaining_genes_parent2):
            child_parent1.insert(position, gene)

        for position, gene in zip(remaining_positions, remaining_genes_parent1):
            child_parent2.insert(position, gene)

        return child_parent1, child_parent2
    
    def mutate_hard(self, solution: List[Tuple[float, float]], 
                   mutation_probability: float, intensity: int = 7) -> List[Tuple[float, float]]:
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
    
    def sort_population(self, population: List[List[Tuple[float, float]]], 
                       fitness: List[float]) -> Tuple[List[List[Tuple[float, float]]], List[float]]:
        combined_lists = list(zip(population, fitness))
        sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])
        sorted_population, sorted_fitness = zip(*sorted_combined_lists)
        return list(sorted_population), list(sorted_fitness)
    
    def validate_population(self, population: List[List[Tuple[float, float]]], 
                          vehicle_data_list: List[Dict[str, Any]] = None) -> List[bool]:
        validation_results = []
        for i, individual in enumerate(population):
            vehicle_data = vehicle_data_list[i] if vehicle_data_list else None
            is_valid = self.restriction_manager.validate_route(individual, vehicle_data)
            validation_results.append(is_valid)
        return validation_results
    
    def get_population_statistics(self, population: List[List[Tuple[float, float]]], 
                                vehicle_data_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        total_individuals = len(population)
        valid_count = 0
        total_penalty = 0.0
        restriction_violations = {}
        
        for i, individual in enumerate(population):
            vehicle_data = vehicle_data_list[i] if vehicle_data_list else None
            summary = self.restriction_manager.get_violation_summary(individual, vehicle_data)
            
            if summary['valid_route']:
                valid_count += 1
            
            total_penalty += summary['total_penalty']
            
            for violation in summary['violations']:
                restriction_name = violation['restriction']
                if restriction_name not in restriction_violations:
                    restriction_violations[restriction_name] = 0
                restriction_violations[restriction_name] += 1
        
        return {
            'total_individuals': total_individuals,
            'valid_individuals': valid_count,
            'invalid_individuals': total_individuals - valid_count,
            'validity_rate': valid_count / total_individuals if total_individuals > 0 else 0,
            'average_penalty': total_penalty / total_individuals if total_individuals > 0 else 0,
            'restriction_violations': restriction_violations,
            'active_restrictions': self.restriction_manager.get_active_restrictions()
        }
