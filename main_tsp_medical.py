#!/usr/bin/env python3
import pygame
from pygame.locals import *
import random
import itertools
import sys
import numpy as np
from core.enhanced_genetic_algorithm import EnhancedGeneticAlgorithm
from core.restriction_manager import RestrictionManager
from core.config_manager import ConfigManager
from restrictions.fuel_restriction import FuelRestriction
from restrictions.route_cost_restriction import RouteCostRestriction
from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction
from restrictions.multiple_vehicles import MultipleVehiclesRestriction
from restrictions.forbidden_routes import ForbiddenRoutes
from restrictions.one_way_routes import OneWayRoutes
from llm.llm_integration import LLMIntegration
from utils.draw_functions import draw_paths, draw_plot, draw_cities
from utils.selection_functions import tournament_or_rank_based_selection
from utils.helper_functions import (
    population_edge_diversity,
    inject_random_individuals,
    inject_heuristic_individuals,
    fast_diversity_aware_selection,
    simple_diversity_aware_mutation
)
from data.benchmark_att48 import att_48_cities_locations, att_48_cities_order
from config.route_cost import route_costs_att_48

class MedicalRouteTSP:
    def __init__(self):
        self.config = ConfigManager()
        
        self.WIDTH = self.config.get("display.width", 1500)
        self.HEIGHT = self.config.get("display.height", 800)
        self.NODE_RADIUS = self.config.get("display.node_radius", 10)
        self.FPS = self.config.get("display.fps", 30)
        self.PLOT_X_OFFSET = self.config.get("display.plot_x_offset", 450)
        
        self.GENERATION_LIMIT = self.config.get("genetic_algorithm.generation_limit", 10000)
        self.N_CITIES = self.config.get("genetic_algorithm.n_cities", 48)
        self.POPULATION_SIZE = self.config.get("genetic_algorithm.population_size", 2000)
        self.N_EXPLORATION_GENERATION = self.config.get("genetic_algorithm.n_exploration_generation", 1000)
        self.N_NEIGHBORS = self.config.get("genetic_algorithm.n_neighbors", 500)
        
        self.INITIAL_MUTATION_PROBABILITY = self.config.get("mutation.initial_probability", 0.85)
        self.INITIAL_MUTATION_INTENSITY = self.config.get("mutation.initial_intensity", 25)
        self.AFTER_EXPLORATION_MUTATION_INTENSITY = self.config.get("mutation.after_exploration_intensity", 5)
        self.AFTER_EXPLORATION_MUTATION_PROBABILITY = self.config.get("mutation.after_exploration_probability", 0.5)
        
        self.diversity_threshold = self.config.get("diversity.threshold", 0.3)
        self.diversity_injection_rate = self.config.get("diversity.injection_rate", 0.1)
        self.diversity_monitor_frequency = self.config.get("diversity.monitor_frequency", 50)
        
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        
        # Algorithm state
        self.generation_without_improvement = 0
        self.finished_exploration = False
        
        # Print loaded configuration
        print(f"Configuration loaded:")
        print(f"- Population size: {self.POPULATION_SIZE}")
        print(f"- Generation limit: {self.GENERATION_LIMIT}")
        print(f"- Display: {self.WIDTH}x{self.HEIGHT}")
        print(f"- Mutation probability: {self.INITIAL_MUTATION_PROBABILITY}")
        print("-" * 50)
        
        self.setup_pygame()
        self.setup_cities()
        self.setup_genetic_algorithm()
        self.setup_restrictions()
        self.setup_llm()
        
        self.generation_counter = itertools.count(start=1)
        self.mutation_intensity = self.INITIAL_MUTATION_INTENSITY
        self.mutation_probability = self.INITIAL_MUTATION_PROBABILITY
        self.best_fitness_values = []
        self.best_solutions = []
        self.last_best_fitness = None
        self.diversity_history = []
        
    def setup_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Medical Route TSP Optimizer - FIAP Tech Challenge")
        self.clock = pygame.time.Clock()
        
    def setup_cities(self):
        att_cities_locations = np.array(att_48_cities_locations)
        max_x = max(point[0] for point in att_cities_locations)
        max_y = max(point[1] for point in att_cities_locations)
        scale_x = (self.WIDTH - self.PLOT_X_OFFSET - self.NODE_RADIUS) / max_x
        scale_y = self.HEIGHT / max_y
        
        self.cities_locations = [
            (int(point[0] * scale_x + self.PLOT_X_OFFSET), int(point[1] * scale_y))
            for point in att_cities_locations
        ]
        
        self.target_solution = [self.cities_locations[i-1] for i in att_48_cities_order]

        print("CITIES LIST FROM PROBLEM", self.cities_locations)
        
    def setup_genetic_algorithm(self):
        self.ga = EnhancedGeneticAlgorithm(self.cities_locations)
        self.fitness_target_solution = self.ga.calculate_base_fitness(self.target_solution)
        print(f"Target Solution Fitness: {self.fitness_target_solution}")
        

    def setup_restrictions(self):
        # Load restriction settings from config
        fuel_config = self.config.get("restrictions.fuel", {})
        capacity_config = self.config.get("restrictions.vehicle_capacity", {})
        hospital_config = self.config.get("restrictions.fixed_start", {})
        route_cost_config = self.config.get("restrictions.route_cost", {})
        multiple_vehicles_config = self.config.get("restrictions.multiple_vehicles", {})
        forbidden_routes_config = self.config.get("restrictions.forbidden_routes", {})
        one_way_routes_config = self.config.get("restrictions.one_way_routes", {})
        
        # Create fuel restriction with config values
        fuel_restriction = FuelRestriction(
            max_distance=fuel_config.get("max_distance", 250.0),
            fuel_cost_per_km=fuel_config.get("fuel_cost_per_km", 0.8),
            fuel_cost_limit=fuel_config.get("fuel_cost_limit", None),
            pixel_to_km_factor=fuel_config.get("pixel_to_km_factor", 0.02)
        )

        fuel_restriction.set_weight(fuel_config.get("weight", 1.0))
        
        # Create vehicle capacity restriction with config values
        capacity_restriction = VehicleCapacityRestriction(
            max_capacity=capacity_config.get("max_capacity", 10),
            delivery_weight_per_city=capacity_config.get("delivery_weight_per_city", 1.0)
        )
        capacity_restriction.set_weight(capacity_config.get("weight", 1.0))

        #Custo da rota
        route_cost_restriction = RouteCostRestriction(
            cities_locations=att_48_cities_locations,
            route_cost_dict=route_costs_att_48,
        )

        route_cost_restriction.set_weight(route_cost_config.get("weight", 1.0))

        # Create fixed start (hospital) restriction
        if hospital_config.get("enabled", False):
            from restrictions.fixed_start_restriction import FixedStartRestriction
            
            # Define primeira cidade como hospital por padrão
            hospital_location = self.cities_locations[0] if hospital_config.get("hospital_is_first_city", True) else None
            
            hospital_restriction = FixedStartRestriction(hospital_location=hospital_location)
            hospital_restriction.set_weight(hospital_config.get("weight", 5.0))
        
        # Create multiple vehicles restriction
        if multiple_vehicles_config.get("enabled", False):
            # Usa a capacidade da restrição de capacidade se disponível, senão usa 1
            vehicle_capacity = capacity_config.get("max_capacity", 1) if capacity_config.get("enabled", False) else 1
            
            multiple_vehicles_restriction = MultipleVehiclesRestriction(
                max_vehicles=multiple_vehicles_config.get("max_vehicles", 5),
                depot=self.cities_locations[0] if hospital_config.get("hospital_is_first_city", True) else None,
                vehicle_capacity=vehicle_capacity
            )
            multiple_vehicles_restriction.set_weight(multiple_vehicles_config.get("weight", 2.0))
            
        # Create forbidden routes restriction
        forbidden_routes_restriction = ForbiddenRoutes(
            base_distance_penalty=forbidden_routes_config.get("base_distance_penalty", 1000.0)
        )
        forbidden_routes_restriction.set_weight(forbidden_routes_config.get("weight", 1.0))
        
        # Create one way routes restriction
        one_way_routes_restriction = OneWayRoutes(
            base_distance_penalty=one_way_routes_config.get("base_distance_penalty", 1000.0)
        )
        one_way_routes_restriction.set_weight(one_way_routes_config.get("weight", 1.0))
        
        # Add restrictions if enabled
        if fuel_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(fuel_restriction)
        
        if capacity_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(capacity_restriction)
        
        if hospital_config.get("enabled", False):
            self.ga.restriction_manager.add_restriction(hospital_restriction)
        
        if route_cost_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(route_cost_restriction)

        if multiple_vehicles_config.get("enabled", False):
            self.ga.restriction_manager.add_restriction(multiple_vehicles_restriction)
            
        if forbidden_routes_config.get("enabled", False):
            self.ga.restriction_manager.add_restriction(forbidden_routes_restriction)
            
        if one_way_routes_config.get("enabled", False):
            self.ga.restriction_manager.add_restriction(one_way_routes_restriction)
        
        print("Active Restrictions:", self.ga.restriction_manager.get_active_restrictions())


    def setup_llm(self):
        llm_config = self.config.get("llm", {})
        if llm_config.get("enabled", True):
            self.llm = LLMIntegration()
            print(f"LLM Integration enabled (fallback mode: {llm_config.get('fallback_mode', True)})")
        else:
            self.llm = None
            print("LLM Integration disabled")
        
    def create_initial_population(self):
        population = self.ga.generate_random_population(self.POPULATION_SIZE - self.N_NEIGHBORS)
        
        for _ in range(self.N_NEIGHBORS):
            nearest_neighbor = self.ga.generate_nearest_neighbor(
                random.randint(0, self.N_CITIES - 1)
            )
            population.append(nearest_neighbor)
            
        return population
        
    def evaluate_population(self, population):
        population_fitness = []
        for individual in population:
            fitness = self.ga.calculate_fitness_with_restrictions(individual)
            population_fitness.append(fitness)
            
        return self.ga.sort_population(population, population_fitness)
        
    def manage_diversity(self, population, current_diversity, generation):
        if current_diversity < self.diversity_threshold and generation % self.diversity_monitor_frequency == 0:
            if random.random() < 0.5:
                print('Injecting random individuals for diversity')
                population = inject_random_individuals(population, self.cities_locations, self.diversity_injection_rate)
            else:
                print('Injecting heuristic individuals for diversity')
                population = inject_heuristic_individuals(population, self.cities_locations, self.diversity_injection_rate)
        return population
        
    def create_new_generation(self, population, population_fitness, current_diversity):
        new_population = [population[0]]  # Elitism
        
        while len(new_population) < self.POPULATION_SIZE:
            parent1, parent2 = fast_diversity_aware_selection(
                population, population_fitness, current_diversity
            )
            
            child1, child2 = self.ga.order_crossover(parent1, parent2)
            
            if current_diversity < self.diversity_threshold:
                print('Low diversity detected - applying aggressive mutation')
                child1 = simple_diversity_aware_mutation(
                    child1, self.mutation_probability, self.mutation_intensity, current_diversity
                )
                child2 = simple_diversity_aware_mutation(
                    child2, self.mutation_probability, self.mutation_intensity, current_diversity
                )
            else:
                child1 = self.ga.mutate_hard(child1, self.mutation_probability, self.mutation_intensity)
                child2 = self.ga.mutate_hard(child2, self.mutation_probability, self.mutation_intensity)
            
            new_population.extend([child1, child2])
            
        return new_population[:self.POPULATION_SIZE]
        
    def update_mutation_parameters(self):
        if self.generation_without_improvement > 100:
            if self.mutation_intensity < self.N_CITIES // 2:
                self.mutation_intensity += 1
                print(f"Increasing mutation intensity to {self.mutation_intensity}")
            
            if self.mutation_probability < 0.9:
                self.mutation_probability += 0.02
                print(f"Increasing mutation probability to {self.mutation_probability:.2f}")
            
            self.generation_without_improvement = 0
            
    def update_exploration_phase(self, generation):
        if generation > self.N_EXPLORATION_GENERATION and not self.finished_exploration:
            print('=== FINISHED EXPLORATION PHASE ===')
            self.mutation_intensity = self.AFTER_EXPLORATION_MUTATION_INTENSITY
            self.mutation_probability = self.AFTER_EXPLORATION_MUTATION_PROBABILITY
            self.finished_exploration = True
            
    def track_progress(self, best_fitness, best_solution):
        if self.finished_exploration:
            if self.last_best_fitness is None or best_fitness < self.last_best_fitness:
                self.last_best_fitness = best_fitness
                self.generation_without_improvement = 0
            else:
                self.generation_without_improvement += 1
                
        self.best_fitness_values.append(best_fitness)
        self.best_solutions.append(best_solution)
        
    def draw_generation(self, population, best_solution):
        self.screen.fill(self.WHITE)
        
        draw_plot(
            self.screen, 
            list(range(len(self.best_fitness_values))), 
            self.best_fitness_values,
            y_label="Fitness - Distance (pixels)"
        )
        
        draw_cities(self.screen, self.cities_locations, self.RED, self.NODE_RADIUS)
        draw_paths(self.screen, best_solution, self.BLUE, width=3)
        
        if len(population) > 1:
            draw_paths(self.screen, population[1], rgb_color=(128, 128, 128), width=1)
            
    def print_generation_info(self, generation, best_fitness, current_diversity, stats):
        # Get logging settings from config
        log_frequency = self.config.get("logging.log_frequency", 100)
        log_generations = self.config.get("logging.log_generations", True)
        
        if log_generations:
            print(f"Generation {generation}: "
                  f"Best fitness = {round(best_fitness, 2)}, "
                  f"Diversity = {current_diversity:.3f}")
        
        if generation % log_frequency == 0:
            print(f"Population Statistics: {stats}")
            
            # Generate LLM instructions every 500 generations
            if self.llm and generation % 500 == 0:
                route_info = {
                    "distance": best_fitness,
                    "restrictions": stats.get('restriction_violations', {}),
                    "generation": generation
                }
                
                instructions = self.llm.generate_delivery_instructions(
                    self.best_solutions[-1], route_info
                )
                print(f"\n=== DELIVERY INSTRUCTIONS (Gen {generation}) ===")
                print(instructions[:300] + "..." if len(instructions) > 300 else instructions)
                print("=" * 50)
                
    def run(self):
        population = self.create_initial_population()
        running = True
        
        print("Starting Medical Route TSP Optimizer...")
        print("Controls:")
        print("  Q - Quit")
        print("  R - Generate route report")
        print("  I - Show route instructions")
        print("  C - Show current configuration")
        print("-" * 50)
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.generate_route_report()
                    elif event.key == pygame.K_i:
                        self.show_route_instructions()
                    elif event.key == pygame.K_c:
                        self.show_current_config()
                        
            generation = next(self.generation_counter)
            
            population, population_fitness = self.evaluate_population(population)
            best_fitness = population_fitness[0]
            best_solution = population[0]
            
            current_diversity = population_edge_diversity(population)
            stats = self.ga.get_population_statistics(population)
            
            self.track_progress(best_fitness, best_solution)
            self.update_exploration_phase(generation)
            population = self.manage_diversity(population, current_diversity, generation)
            
            population = self.create_new_generation(population, population_fitness, current_diversity)
            
            self.update_mutation_parameters()
            self.draw_generation(population, best_solution)
            self.print_generation_info(generation, best_fitness, current_diversity, stats)
            
            pygame.display.flip()
            self.clock.tick(self.FPS)
            
            if generation >= self.GENERATION_LIMIT:
                print('Generation limit reached. Stopping algorithm')
                self.final_report()
                running = False
                
        pygame.quit()
        sys.exit()
        
    def generate_route_report(self):
        if not self.best_solutions:
            return
            
        routes_data = []
        for i, solution in enumerate(self.best_solutions[-10:], 1):
            fitness = self.ga.calculate_fitness_with_restrictions(solution)
            routes_data.append({
                'route_id': i,
                'distance': fitness,
                'time': fitness / 50,
                'efficiency': max(0, 100 - (fitness / self.fitness_target_solution * 100 - 100)),
                'violations': []
            })
        
        if self.llm:
            report = self.llm.generate_route_report(routes_data, "recent")
            print(f"\n=== ROUTE PERFORMANCE REPORT ===")
            print(report)
            print("=" * 50)
        else:
            print("\n=== ROUTE PERFORMANCE REPORT ===")
            print("LLM disabled. Basic statistics:")
            print(f"Routes analyzed: {len(routes_data)}")
            print(f"Average fitness: {sum(r['distance'] for r in routes_data) / len(routes_data):.2f}")
            print("=" * 50)
        
    def show_route_instructions(self):
        if not self.best_solutions:
            return
            
        best_solution = self.best_solutions[-1]
        best_fitness = self.best_fitness_values[-1]
        
        route_info = {
            "distance": best_fitness,
            "restrictions": self.ga.restriction_manager.get_active_restrictions(),
            "is_valid": self.ga.restriction_manager.validate_route(best_solution)
        }
        
        if self.llm:
            instructions = self.llm.generate_delivery_instructions(best_solution, route_info)
            print(f"\n=== CURRENT BEST ROUTE INSTRUCTIONS ===")
            print(instructions)
            print("=" * 50)
        else:
            print(f"\n=== CURRENT BEST ROUTE INSTRUCTIONS ===")
            print("LLM disabled. Basic route info:")
            print(f"Distance: {best_fitness:.2f} pixels")
            print(f"Cities: {len(best_solution)}")
            print(f"Valid route: {route_info['is_valid']}")
            print("=" * 50)
    
    def show_current_config(self):
        print(f"\n=== CURRENT CONFIGURATION ===")
        print(f"Generation limit: {self.GENERATION_LIMIT}")
        print(f"Population size: {self.POPULATION_SIZE}")
        print(f"Display: {self.WIDTH}x{self.HEIGHT}")
        print(f"Mutation probability: {self.INITIAL_MUTATION_PROBABILITY}")
        print(f"Active restrictions: {self.ga.restriction_manager.get_active_restrictions()}")
        
        # Show restriction details
        for restriction in self.ga.restriction_manager.restrictions:
            if restriction.name == "fuel_restriction":
                fuel_restriction = restriction
                fuel_info = fuel_restriction.get_fuel_consumption([(0,0), (100,0)])
                print(f"Fuel restriction: max {fuel_restriction.max_distance}km, cost {fuel_restriction.fuel_cost_per_km}/km")
                print(f"Fuel Cost Limit: R$ {fuel_restriction.fuel_cost_limit}")
            elif restriction.name == "vehicle_capacity_restriction":
                capacity_restriction = restriction
                print(f"Vehicle capacity: max {capacity_restriction.max_capacity} deliveries")
            elif restriction.name == "route_cost_restriction":
                route_cost_restriction = restriction
                print(f"Route cost: {route_cost_restriction.route_cost_dict}")
            elif restriction.name == "multiple_vehicles_restriction":
                multiple_vehicles_restriction = restriction
                print(f"Multiple vehicles: max {multiple_vehicles_restriction.max_vehicles} vehicles, capacity {multiple_vehicles_restriction.vehicle_capacity} patients/vehicle, depot at {multiple_vehicles_restriction.depot}")
            elif restriction.name == "forbidden_routes":
                forbidden_routes_restriction = restriction
                print(f"Forbidden routes: {len(forbidden_routes_restriction.get_all_forbidden_routes())} routes, penalty {forbidden_routes_restriction._base_distance_penalty}")
            elif restriction.name == "one_way_routes":
                one_way_routes_restriction = restriction
                print(f"One-way routes: {len(one_way_routes_restriction.get_all_one_way_routes())} routes, penalty {one_way_routes_restriction._base_distance_penalty}")
        
        print("=" * 50)
        
    def final_report(self):
        print(f"\n=== FINAL OPTIMIZATION REPORT ===")
        print(f"Total generations: {len(self.best_fitness_values)}")
        print(f"Best fitness achieved: {min(self.best_fitness_values):.2f}")
        print(f"Target fitness: {self.fitness_target_solution:.2f}")
        
        improvement = ((self.fitness_target_solution - min(self.best_fitness_values)) / 
                      self.fitness_target_solution * 100)
        print(f"Improvement over target: {improvement:.1f}%")
        
        final_stats = self.ga.get_population_statistics([self.best_solutions[-1]])
        print(f"Final solution statistics: {final_stats}")
        
        print(f"\nConfiguration used:")
        print(f"- Population size: {self.POPULATION_SIZE}")
        print(f"- Generation limit: {self.GENERATION_LIMIT}")
        print(f"- Restrictions: {self.ga.restriction_manager.get_active_restrictions()}")

if __name__ == "__main__":
    optimizer = MedicalRouteTSP()
    optimizer.run()