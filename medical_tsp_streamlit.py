#!/usr/bin/env python3
import streamlit as st
import plotly.graph_objects as go
import time
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
from llm.llm_integration import LLMIntegration
# from utils.draw_functions import draw_paths, draw_plot, draw_cities  # pygame specific
# from utils.selection_functions import tournament_or_rank_based_selection  # not used
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
        
        # self.setup_pygame()  # removed for streamlit
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
        
    # def setup_pygame(self):  # removed for streamlit
    #     pygame.init()
    #     self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
    #     pygame.display.set_caption("Medical Route TSP Optimizer - FIAP Tech Challenge")
    #     self.clock = pygame.time.Clock()
        
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
            max_patients=capacity_config.get("max_patients", 10),
            vehicle_data={'city_patients': {'1352_224': 1, '1423_195': 1}}
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
            hospital_restriction = FixedStartRestriction()
            
            # Define primeira cidade como hospital
            if hospital_config.get("hospital_is_first_city", True):
                hospital_restriction.set_hospital_location(self.cities_locations[0])
            
            hospital_restriction.set_weight(hospital_config.get("weight", 5.0))
        
        # Create multiple vehicles restriction
        if multiple_vehicles_config.get("enabled", False):
            # Usa a capacidade da restrição de capacidade se disponível, senão usa 1
            vehicle_capacity = capacity_config.get("max_patients", 1) if capacity_config.get("enabled", False) else 1
            
            multiple_vehicles_restriction = MultipleVehiclesRestriction(
                max_vehicles=multiple_vehicles_config.get("max_vehicles", 5),
                depot=self.cities_locations[0] if hospital_config.get("hospital_is_first_city", True) else None,
                vehicle_capacity=vehicle_capacity
            )
            multiple_vehicles_restriction.set_weight(multiple_vehicles_config.get("weight", 2.0))
        
        # Add restrictions if enabled
        if fuel_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(fuel_restriction)
        
        if capacity_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(capacity_restriction)
        
        if hospital_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(hospital_restriction)
        
        if route_cost_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(route_cost_restriction)

        if multiple_vehicles_config.get("enabled", True):
            self.ga.restriction_manager.add_restriction(multiple_vehicles_restriction)
        
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
        
    # def draw_generation(self, population, best_solution):  # removed for streamlit
    #     self.screen.fill(self.WHITE)
    #     
    #     draw_plot(
    #         self.screen, 
    #         list(range(len(self.best_fitness_values))), 
    #         self.best_fitness_values,
    #         y_label="Fitness - Distance (pixels)"
    #     )
    #     
    #     draw_cities(self.screen, self.cities_locations, self.RED, self.NODE_RADIUS)
    #     draw_paths(self.screen, best_solution, self.BLUE, width=3)
    #     
    #     if len(population) > 1:
    #         draw_paths(self.screen, population[1], rgb_color=(128, 128, 128), width=1)
            
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
                
    # run() method replaced by streamlit frontend below
        
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
                print(f"Vehicle capacity: max {capacity_restriction.max_patients} deliveries")
            elif restriction.name == "route_cost_restriction":
                route_cost_restriction = restriction
                print(f"Route cost: {route_cost_restriction.route_cost_dict}")
            elif restriction.name == "multiple_vehicles_restriction":
                multiple_vehicles_restriction = restriction
                print(f"Multiple vehicles: max {multiple_vehicles_restriction.max_vehicles} vehicles, capacity {multiple_vehicles_restriction.vehicle_capacity} patients/vehicle, depot at {multiple_vehicles_restriction.depot}")
        
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

# ===============================
# STREAMLIT FRONTEND
# ===============================

# Page config
st.set_page_config(page_title="Medical TSP GA", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if 'generation' not in st.session_state:
    st.session_state.optimizer = MedicalRouteTSP()
    st.session_state.generation = 0
    st.session_state.population = st.session_state.optimizer.create_initial_population()

optimizer = st.session_state.optimizer

# Header
st.title("Medical Route TSP Optimizer - FIAP Tech Challenge")
st.markdown("**Algoritmo Genético com Restrições Médicas**")

# Info metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Geração Atual", f"{st.session_state.generation}/{optimizer.GENERATION_LIMIT}")
with col2:
    if optimizer.best_fitness_values:
        st.metric("Melhor Fitness", f"{round(optimizer.best_fitness_values[-1], 2)}")
    else:
        st.metric("Melhor Fitness", "---")
with col3:
    st.metric("População", optimizer.POPULATION_SIZE)


st.divider()

# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    next_gen = st.button("Próxima Geração", use_container_width=True, type="primary")
with col2:
    run_all = st.button("Executar Todas", use_container_width=True)

st.divider()

# Current configuration - small and discrete, always visible
st.caption("Configuração:")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"População: {optimizer.POPULATION_SIZE} | Gerações: {optimizer.GENERATION_LIMIT}")
with col2:
    st.caption(f"Mutação: {optimizer.INITIAL_MUTATION_PROBABILITY} | Restrições: {len(optimizer.ga.restriction_manager.get_active_restrictions())}")
with col3:
    # Show key restrictions briefly
    restrictions_summary = []
    for restriction in optimizer.ga.restriction_manager.restrictions:
        if restriction.name == "fuel_restriction":
            restrictions_summary.append(f"Combustível: {restriction.max_distance}km")
        elif restriction.name == "vehicle_capacity_restriction":
            restrictions_summary.append(f"Capacidade: {restriction.max_patients}")
    if restrictions_summary:
        st.caption(" | ".join(restrictions_summary[:2]))

# Main layout: Map full width, Chart below
st.subheader("1. Mapa de Rotas")
map_placeholder = st.empty()

st.subheader("2. Evolução do Fitness")
fitness_placeholder = st.empty()

def create_map(best_solution, population):
    """Helper function to create the map figure"""
    fig = go.Figure()
    
    # Add best route
    if best_solution:
        route_x = [x for x, y in best_solution] + [best_solution[0][0]]
        route_y = [y for x, y in best_solution] + [best_solution[0][1]]
        
        fig.add_trace(go.Scatter(
            x=route_x,
            y=route_y,
            mode='lines+markers',
            line=dict(width=3, color='blue'),
            marker=dict(size=8, color='red'),
            name='Melhor Rota',
        ))
        
        # Add second best route
        if len(population) > 1:
            second_best = population[1]
            second_x = [x for x, y in second_best] + [second_best[0][0]]
            second_y = [y for x, y in second_best] + [second_best[0][1]]
            
            fig.add_trace(go.Scatter(
                x=second_x,
                y=second_y,
                mode='lines',
                line=dict(width=1.5, color='lightgray'),
                name='2ª Melhor Rota',
            ))
    
    # Add city markers
    cities_x = [x for x, y in optimizer.cities_locations]
    cities_y = [y for x, y in optimizer.cities_locations]
    
    fig.add_trace(go.Scatter(
        x=cities_x,
        y=cities_y,
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Cidades'
    ))
    
    fig.update_layout(
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)"
    )
    
    return fig

# Display initial or current state
current_best = optimizer.best_solutions[-1] if optimizer.best_solutions else None
current_pop = st.session_state.population if st.session_state.population else []

fig = create_map(current_best, current_pop)
map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{st.session_state.generation}")

if optimizer.best_fitness_values:
    fitness_placeholder.line_chart(optimizer.best_fitness_values, height=300)

# Execute next generation - SEGUINDO EXATAMENTE A LÓGICA DO main_tsp_medical.py
if next_gen and st.session_state.generation < optimizer.GENERATION_LIMIT:
    generation = next(optimizer.generation_counter)
    st.session_state.generation = generation
    
    # Essa é a lógica EXATA do main_tsp_medical.py
    st.session_state.population, population_fitness = optimizer.evaluate_population(st.session_state.population)
    best_fitness = population_fitness[0]
    best_solution = st.session_state.population[0]
    
    current_diversity = population_edge_diversity(st.session_state.population)
    stats = optimizer.ga.get_population_statistics(st.session_state.population)
    
    optimizer.track_progress(best_fitness, best_solution)
    optimizer.update_exploration_phase(generation)
    st.session_state.population = optimizer.manage_diversity(st.session_state.population, current_diversity, generation)
    
    st.session_state.population = optimizer.create_new_generation(st.session_state.population, population_fitness, current_diversity)
    
    optimizer.update_mutation_parameters()
    optimizer.print_generation_info(generation, best_fitness, current_diversity, stats)
    
    if generation >= optimizer.GENERATION_LIMIT:
        st.success('Generation limit reached. Stopping algorithm')
        optimizer.final_report()
    
    st.rerun()

# Run all generations
if run_all and st.session_state.generation < optimizer.GENERATION_LIMIT:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(optimizer.GENERATION_LIMIT - st.session_state.generation):
        generation = next(optimizer.generation_counter)
        st.session_state.generation = generation
        
        # Lógica EXATA do main_tsp_medical.py
        st.session_state.population, population_fitness = optimizer.evaluate_population(st.session_state.population)
        best_fitness = population_fitness[0]
        best_solution = st.session_state.population[0]
        
        current_diversity = population_edge_diversity(st.session_state.population)
        stats = optimizer.ga.get_population_statistics(st.session_state.population)
        
        optimizer.track_progress(best_fitness, best_solution)
        optimizer.update_exploration_phase(generation)
        st.session_state.population = optimizer.manage_diversity(st.session_state.population, current_diversity, generation)
        
        status_text.info(f"Geração {generation}/{optimizer.GENERATION_LIMIT} | Melhor Fitness: {round(best_fitness, 2)}")
        progress_bar.progress(generation / optimizer.GENERATION_LIMIT)
        
        # Update visualizations
        fig = create_map(best_solution, st.session_state.population)
        map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_run_{generation}")
        
        if optimizer.best_fitness_values:
            fitness_placeholder.line_chart(optimizer.best_fitness_values, height=300)
        
        st.session_state.population = optimizer.create_new_generation(st.session_state.population, population_fitness, current_diversity)
        
        optimizer.update_mutation_parameters()
        optimizer.print_generation_info(generation, best_fitness, current_diversity, stats)
        
        if generation >= optimizer.GENERATION_LIMIT:
            optimizer.final_report()
            break
            
        time.sleep(0.1)
    
    progress_bar.empty()
    #status_text.success(f"Simulação concluída!")
    st.rerun()

# Relatórios section
if optimizer.best_solutions:
    st.divider()
    st.subheader("3. Relatórios")
    
    # 3.1 Ver detalhes da rota atual
    with st.expander("3.1 Detalhes da Rota Atual", expanded=False):
        last_route = optimizer.best_solutions[-1]
        for idx, point in enumerate(last_route, 1):
            st.write(f"{idx}. Posição: {point}")
    
    # 3.2 Relatório de Performance das Rotas
    with st.expander("3.2 Relatório de Performance", expanded=False):
        # Generate report data
        routes_data = []
        for i, solution in enumerate(optimizer.best_solutions[-10:], 1):
            fitness = optimizer.ga.calculate_fitness_with_restrictions(solution)
            routes_data.append({
                'route_id': i,
                'distance': fitness,
                'time': fitness / 50,
                'efficiency': max(0, 100 - (fitness / optimizer.fitness_target_solution * 100 - 100)),
                'violations': []
            })
        
        # Create report
        total_distance = sum(r['distance'] for r in routes_data)
        avg_efficiency = sum(r['efficiency'] for r in routes_data) / len(routes_data)
        
        report = f"""=== RELATÓRIO DE PERFORMANCE DAS ROTAS MÉDICAS ===

RESUMO EXECUTIVO:
- Total de rotas executadas: {len(routes_data)}
- Distância total percorrida: {total_distance:.2f} pixels
- Eficiência média: {avg_efficiency:.1f}%
- Taxa de problemas: 0.0%

ANÁLISE DE PERFORMANCE:
O sistema de otimização está funcionando adequadamente. 
Recomenda-se monitoramento contínuo para identificar oportunidades de melhoria.

PRÓXIMOS PASSOS:
1. Configurar integração completa com LLM
2. Implementar alertas automáticos
3. Expandir coleta de dados de performance

OBSERVAÇÃO: Relatório em modo fallback. Configure LLM para análises detalhadas."""
        
        st.code(report)
    
    # 3.3 Instruções da Rota
    with st.expander("3.3 Instruções da Rota", expanded=False):
        best_solution = optimizer.best_solutions[-1]
        best_fitness = optimizer.best_fitness_values[-1]
        
        route_info = {
            "distance": best_fitness,
            "restrictions": optimizer.ga.restriction_manager.get_active_restrictions(),
            "is_valid": optimizer.ga.restriction_manager.validate_route(best_solution)
        }
        
        if optimizer.llm:
            try:
                instructions = optimizer.llm.generate_delivery_instructions(best_solution, route_info)
                st.code(instructions)
            except:
                st.code(f"""=== INSTRUÇÕES DA ROTA ===
Distância: {best_fitness:.2f} pixels
Cidades: {len(best_solution)}
Rota válida: {route_info['is_valid']}
Restrições ativas: {route_info['restrictions']}""")
        else:
            st.code(f"""=== INSTRUÇÕES DA ROTA ===
Distância: {best_fitness:.2f} pixels
Cidades: {len(best_solution)}
Rota válida: {route_info['is_valid']}
Restrições ativas: {route_info['restrictions']}""")

# Final message when simulation is complete
if st.session_state.generation >= optimizer.GENERATION_LIMIT:
    st.write("**Simulação concluída!**")
    
    # 3.4 Estatísticas Finais (only shown when complete)
    with st.expander("3.4 Estatísticas Finais", expanded=True):
        if optimizer.best_solutions:
            final_stats = optimizer.ga.get_population_statistics([optimizer.best_solutions[-1]])
            st.code(f"Population Statistics: {final_stats}")
            
            best_fitness = min(optimizer.best_fitness_values) if optimizer.best_fitness_values else 0
            target_fitness = optimizer.fitness_target_solution
            improvement = ((target_fitness - best_fitness) / target_fitness * 100)
            
            final_report = f"""=== RELATÓRIO FINAL DE OTIMIZAÇÃO ===
Total de gerações: {len(optimizer.best_fitness_values)}
Melhor fitness alcançado: {best_fitness:.2f}
Fitness alvo: {target_fitness:.2f}
Melhoria sobre o alvo: {improvement:.1f}%
Estatísticas da solução final: {final_stats}

Configuração utilizada:
- Tamanho da população: {optimizer.POPULATION_SIZE}
- Limite de gerações: {optimizer.GENERATION_LIMIT}
- Restrições: {optimizer.ga.restriction_manager.get_active_restrictions()}"""
            
            st.code(final_report)
    
    # 3.5 Configuração Detalhada
    with st.expander("3.5 Configuração Detalhada", expanded=False):
        config_info = f"""=== CONFIGURAÇÃO COMPLETA ===
Limite de gerações: {optimizer.GENERATION_LIMIT}
Tamanho da população: {optimizer.POPULATION_SIZE}
Display: {optimizer.WIDTH}x{optimizer.HEIGHT}
Probabilidade de mutação: {optimizer.INITIAL_MUTATION_PROBABILITY}
Restrições ativas: {optimizer.ga.restriction_manager.get_active_restrictions()}"""
        
        # Show restriction details
        for restriction in optimizer.ga.restriction_manager.restrictions:
            if restriction.name == "fuel_restriction":
                fuel_restriction = restriction
                config_info += f"\nRestrição de combustível: max {fuel_restriction.max_distance}km, custo {fuel_restriction.fuel_cost_per_km}/km"
                config_info += f"\nLimite de custo de combustível: R$ {fuel_restriction.fuel_cost_limit}"
            elif restriction.name == "vehicle_capacity_restriction":
                capacity_restriction = restriction
                config_info += f"\nCapacidade do veículo: max {capacity_restriction.max_patients} entregas"
            elif restriction.name == "route_cost_restriction":
                route_cost_restriction = restriction
                config_info += f"\nCusto da rota: {route_cost_restriction.route_cost_dict}"
            elif restriction.name == "multiple_vehicles_restriction":
                multiple_vehicles_restriction = restriction
                config_info += f"\nMúltiplos veículos: max {multiple_vehicles_restriction.max_vehicles} veículos, capacidade {multiple_vehicles_restriction.vehicle_capacity} pacientes/veículo, depósito em {multiple_vehicles_restriction.depot}"
        
        config_info += "\n" + "=" * 50
        
        st.code(config_info)