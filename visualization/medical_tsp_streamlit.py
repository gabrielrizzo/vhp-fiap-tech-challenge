import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

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
from utils.helper_functions import (
    population_edge_diversity,
    inject_random_individuals,
    inject_heuristic_individuals,
    fast_diversity_aware_selection,
    simple_diversity_aware_mutation
)
from data.benchmark_att48 import att_48_cities_locations, att_48_cities_order
from data.benchmark_hospitals_sp import hospitals_sp_data
from config.route_cost import route_costs_att_48
 
class MedicalRouteTSP:
    def __init__(self, dataset_type='att48', sidebar_config=None):
        self.config = ConfigManager()
        self.dataset_type = dataset_type
        self.sidebar_config = sidebar_config  # Armazena configurações do sidebar ANTES de setup
 
        self.WIDTH = self.config.get("display.width", 1500)
        self.HEIGHT = self.config.get("display.height", 800)
        self.NODE_RADIUS = self.config.get("display.node_radius", 10)
        self.FPS = self.config.get("display.fps", 30)
        self.PLOT_X_OFFSET = self.config.get("display.plot_x_offset", 450)
 
        self.GENERATION_LIMIT = self.config.get("genetic_algorithm.generation_limit", 10000)
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
 
        self.generation_without_improvement = 0
        self.finished_exploration = False
 
        print(f"Configuration loaded:")
        print(f"- Dataset: {dataset_type}")
        print(f"- Population size: {self.POPULATION_SIZE}")
        print(f"- Generation limit: {self.GENERATION_LIMIT}")
        print(f"- Display: {self.WIDTH}x{self.HEIGHT}")
        print(f"- Mutation probability: {self.INITIAL_MUTATION_PROBABILITY}")
        print("-" * 50)
 
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
 
    def setup_cities(self):
        if self.dataset_type == 'hospitals_sp':
            # Hospitais de São Paulo (coordenadas geográficas)
            self.cities_locations = [(d['lat'], d['lon']) for d in hospitals_sp_data]
            self.city_names = [d['name'] for d in hospitals_sp_data]
            self.N_CITIES = len(self.cities_locations)
            self.target_solution = None
            self.use_mapbox = True
            print(f"Using Hospitals SP dataset: {self.N_CITIES} locations")
        else:
            # ATT48 (coordenadas em pixels)
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
            self.city_names = [f"City {i+1}" for i in range(len(self.cities_locations))]
            self.N_CITIES = len(self.cities_locations)
            self.use_mapbox = False
            print(f"Using ATT48 dataset: {self.N_CITIES} cities")
 
        print("CITIES LIST FROM PROBLEM", self.cities_locations[:3], "...")
 
    def setup_genetic_algorithm(self):
        self.ga = EnhancedGeneticAlgorithm(self.cities_locations)
        if self.target_solution:
            self.fitness_target_solution = self.ga.calculate_base_fitness(self.target_solution)
            print(f"Target Solution Fitness: {self.fitness_target_solution}")
        else:
            self.fitness_target_solution = None
            print("No target solution for this dataset")
 
    def setup_restrictions(self):
        # IMPORTANTE: Limpa todas as restrições antes de adicionar novas
        self.ga.restriction_manager.restrictions = []
 
        fuel_config = self.config.get("restrictions.fuel", {})
        capacity_config = self.config.get("restrictions.vehicle_capacity", {})
        hospital_config = self.config.get("restrictions.fixed_start", {})
        route_cost_config = self.config.get("restrictions.route_cost", {})
        multiple_vehicles_config = self.config.get("restrictions.multiple_vehicles", {})
 
        # Usa configurações do sidebar se disponíveis, senão usa config.json
        if hasattr(self, 'sidebar_config') and self.sidebar_config:
            # Sidebar tem prioridade total
            fuel_enabled = self.sidebar_config.get('fuel_enabled', False)
            capacity_enabled = self.sidebar_config.get('capacity_enabled', False)
            fixed_start_enabled = self.sidebar_config.get('fixed_start_enabled', False)
            route_cost_enabled = self.sidebar_config.get('route_cost_enabled', False)
            multiple_vehicles_enabled = self.sidebar_config.get('multiple_vehicles_enabled', False)
 
            fuel_max_distance = self.sidebar_config.get('fuel_max_distance', 250.0)
            fuel_cost_per_km = self.sidebar_config.get('fuel_cost_per_km', 0.8)
            fuel_cost_limit = self.sidebar_config.get('fuel_cost_limit', 300.0)
            max_patients = self.sidebar_config.get('max_capacity', 10)
            max_vehicles_count = self.sidebar_config.get('max_vehicles', 5)
 
            print(f"DEBUG - Using sidebar config:")
            print(f"  fuel_enabled: {fuel_enabled}")
            print(f"  capacity_enabled: {capacity_enabled}")
            print(f"  fixed_start_enabled: {fixed_start_enabled}")
            print(f"  route_cost_enabled: {route_cost_enabled}")
            print(f"  multiple_vehicles_enabled: {multiple_vehicles_enabled}")
        else:
            # Usa config.json apenas se não houver sidebar_config
            fuel_enabled = fuel_config.get("enabled", True)
            capacity_enabled = capacity_config.get("enabled", True)
            fixed_start_enabled = hospital_config.get("enabled", True)
            route_cost_enabled = route_cost_config.get("enabled", True)
            multiple_vehicles_enabled = multiple_vehicles_config.get("enabled", True)
 
            fuel_max_distance = fuel_config.get("max_distance", 250.0)
            fuel_cost_per_km = fuel_config.get("fuel_cost_per_km", 0.8)
            fuel_cost_limit = fuel_config.get("fuel_cost_limit", None)
            max_patients = capacity_config.get("max_capacity", 10)
            max_vehicles_count = multiple_vehicles_config.get("max_vehicles", 5)
 
            print(f"DEBUG - Using config.json (no sidebar)")
 
        # Cria restrições apenas se estiverem habilitadas
        if fuel_enabled:
            fuel_restriction = FuelRestriction(
                max_distance=fuel_max_distance,
                fuel_cost_per_km=fuel_cost_per_km,
                fuel_cost_limit=fuel_cost_limit,
                pixel_to_km_factor=fuel_config.get("pixel_to_km_factor", 0.02)
            )
            fuel_restriction.set_weight(fuel_config.get("weight", 1.0))
            self.ga.restriction_manager.add_restriction(fuel_restriction)
            print("  ✓ Fuel restriction added")
 
        if capacity_enabled:
            capacity_restriction = VehicleCapacityRestriction(
                max_patients_per_vehicle=max_patients
            )
            capacity_restriction.set_weight(capacity_config.get("weight", 1.0))
            self.ga.restriction_manager.add_restriction(capacity_restriction)
            print("  ✓ Capacity restriction added")
 
        if route_cost_enabled and self.dataset_type == 'att48':
            route_cost_restriction = RouteCostRestriction(
                cities_locations=att_48_cities_locations,
                route_cost_dict=route_costs_att_48,
            )
            route_cost_restriction.set_weight(route_cost_config.get("weight", 1.0))
            self.ga.restriction_manager.add_restriction(route_cost_restriction)
            print("  ✓ Route cost restriction added")
 
        if fixed_start_enabled:
            from restrictions.fixed_start_restriction import FixedStartRestriction
            hospital_restriction = FixedStartRestriction()
 
            if hospital_config.get("hospital_is_first_city", True):
                hospital_restriction.set_hospital_location(self.cities_locations[0])
 
            hospital_restriction.set_weight(hospital_config.get("weight", 5.0))
            self.ga.restriction_manager.add_restriction(hospital_restriction)
            print("  ✓ Fixed start restriction added")
 
        if multiple_vehicles_enabled:
            vehicle_capacity = max_patients if capacity_enabled else 1
 
            multiple_vehicles_restriction = MultipleVehiclesRestriction(
                max_vehicles=max_vehicles_count,
                depot=self.cities_locations[0] if fixed_start_enabled else None,
                vehicle_capacity=vehicle_capacity
            )
            multiple_vehicles_restriction.set_weight(multiple_vehicles_config.get("weight", 2.0))
            self.ga.restriction_manager.add_restriction(multiple_vehicles_restriction)
            print("  ✓ Multiple vehicles restriction added")
 
        print(f"\nActive Restrictions: {self.ga.restriction_manager.get_active_restrictions()}")
        print(f"Total restrictions added: {len(self.ga.restriction_manager.restrictions)}\n")
 
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
 
    def evaluate_population(self, population, vehicle_data_list=None):
        population_fitness = []
        for i, individual in enumerate(population):
            vehicle_data = vehicle_data_list[i] if vehicle_data_list else None
            fitness = self.ga.calculate_fitness_with_restrictions(individual, vehicle_data)
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
        new_population = [population[0]]
 
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
 
    def print_generation_info(self, generation, best_fitness, current_diversity, stats):
        log_frequency = self.config.get("logging.log_frequency", 100)
        log_generations = self.config.get("logging.log_generations", True)
 
        if log_generations:
            print(f"Generation {generation}: "
                  f"Best fitness = {round(best_fitness, 2)}, "
                  f"Diversity = {current_diversity:.3f}")
 
        if generation % log_frequency == 0:
            print(f"Population Statistics: {stats}")
 
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
 
    def final_report(self):
        print(f"\n=== FINAL OPTIMIZATION REPORT ===")
        print(f"Total generations: {len(self.best_fitness_values)}")
        print(f"Best fitness achieved: {min(self.best_fitness_values):.2f}")
        if self.fitness_target_solution:
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
 
st.set_page_config(page_title="Medical TSP GA", layout="wide", initial_sidebar_state="expanded")
 
# Sidebar para escolher dataset e configurar restrições
with st.sidebar:
    st.header("⚙️ Configurações")
 
    # Dataset selection
    dataset_choice = st.selectbox(
        "Escolha o Dataset:",
        options=['att48', 'hospitals_sp'],
        format_func=lambda x: "🏙️ ATT48 Benchmark" if x == 'att48' else "🏥 Hospitais de São Paulo",
        index=0,
        help="ATT48: Benchmark clássico | Hospitais SP: Dados reais com mapa"
    )
 
    st.divider()
 
    # Restrictions configuration
    st.subheader("🚧 Restrições")
 
    # Fuel Restriction
    fuel_enabled = st.checkbox("Combustível", value=True, help="Limita distância máxima e custo de combustível")
    fuel_max_distance = 250.0
    fuel_cost_per_km = 0.8
    fuel_cost_limit = 300.0
    if fuel_enabled:
        with st.expander("⚙️ Configurar Combustível"):
            fuel_max_distance = st.number_input("Distância Máxima (km)", min_value=50.0, max_value=500.0, value=250.0, step=10.0)
            fuel_cost_per_km = st.number_input("Custo por km (R$)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)
            fuel_cost_limit = st.number_input("Limite de Custo (R$)", min_value=0.0, max_value=1000.0, value=300.0, step=10.0)
 
    # Vehicle Capacity Restriction
    capacity_enabled = st.checkbox("Capacidade do Veículo", value=True, help="Limita número de pacientes por veículo")
    max_patients = 10
    if capacity_enabled:
        with st.expander("⚙️ Configurar Capacidade"):
            max_patients = st.slider("Pacientes por Veículo", min_value=1, max_value=20, value=10, step=1)
 
    # Fixed Start Restriction
    fixed_start_enabled = st.checkbox("Início Fixo (Hospital)", value=True, help="Força rota começar no hospital")
 
    # Route Cost Restriction
    route_cost_enabled = st.checkbox("Custo de Rotas", value=True, help="Adiciona custos específicos para certas rotas (pedágios, etc)")
 
    # Multiple Vehicles Restriction
    multiple_vehicles_enabled = st.checkbox("Múltiplos Veículos", value=True, help="Permite distribuir pacientes entre várias ambulâncias")
    max_vehicles = 5
    if multiple_vehicles_enabled:
        with st.expander("⚙️ Configurar Veículos"):
            max_vehicles = st.slider("Número Máximo de Veículos", min_value=1, max_value=10, value=5, step=1)
 
    st.divider()
 
    # Botão para aplicar configurações
    apply_config = st.button("🔄 Aplicar Configurações", use_container_width=True, type="primary")
 
    st.divider()
 
    # Summary of active restrictions
    active_count = sum([fuel_enabled, capacity_enabled, fixed_start_enabled, route_cost_enabled, multiple_vehicles_enabled])
    st.caption(f"✅ {active_count} restrições ativas")
 
    if dataset_choice == 'att48':
        st.caption("**ATT48 Benchmark**: 48 cidades em pixels com solução ótima conhecida")
    else:
        st.caption("**Hospitais SP**: 48 hospitais reais com mapa interativo")
 
# Initialize session state
if 'generation' not in st.session_state or st.session_state.get('dataset_type') != dataset_choice or apply_config:
    # Criar dicionário de configurações do sidebar
    sidebar_config = {}
    if fuel_enabled:
        sidebar_config.update({
            'fuel_enabled': fuel_enabled,
            'fuel_max_distance': fuel_max_distance if fuel_enabled else 250.0,
            'fuel_cost_per_km': fuel_cost_per_km if fuel_enabled else 0.8,
            'fuel_cost_limit': fuel_cost_limit if fuel_enabled else 300.0,
        })
    else:
        sidebar_config['fuel_enabled'] = False
 
    if capacity_enabled:
        sidebar_config.update({
            'capacity_enabled': capacity_enabled,
            'max_capacity': max_patients if capacity_enabled else 10,
        })
    else:
        sidebar_config['capacity_enabled'] = False
 
    sidebar_config['fixed_start_enabled'] = fixed_start_enabled
    sidebar_config['route_cost_enabled'] = route_cost_enabled
 
    if multiple_vehicles_enabled:
        sidebar_config.update({
            'multiple_vehicles_enabled': multiple_vehicles_enabled,
            'max_vehicles': max_vehicles if multiple_vehicles_enabled else 5,
        })
    else:
        sidebar_config['multiple_vehicles_enabled'] = False
 
    # Criar optimizer COM sidebar_config no construtor (evita duplicação)
    st.session_state.optimizer = MedicalRouteTSP(dataset_type=dataset_choice, sidebar_config=sidebar_config)
    st.session_state.generation = 0
    st.session_state.population = st.session_state.optimizer.create_initial_population()
    st.session_state.dataset_type = dataset_choice
 
    if apply_config:
        st.success("✅ Configurações aplicadas! População resetada.")
        st.rerun()
 
optimizer = st.session_state.optimizer
 
# Header
st.title("🧬 Medical Route TSP Optimizer - FIAP Tech Challenge")
dataset_name = "ATT48 Benchmark" if dataset_choice == 'att48' else "Hospitais de São Paulo"
st.markdown(f"**Algoritmo Genético com Restrições Médicas - Dataset: {dataset_name}**")
 
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
with col4:
    phase = "Exploração" if not optimizer.finished_exploration else "Refinamento"
    st.metric("Fase", phase)
 
st.divider()
 
# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    next_gen = st.button("▶️ Próxima Geração", use_container_width=True, type="primary")
with col2:
    run_all = st.button("⏩ Executar Todas", use_container_width=True)
 
st.divider()
 
# Current configuration
st.caption("Configuração:")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"População: {optimizer.POPULATION_SIZE} | Gerações: {optimizer.GENERATION_LIMIT}")
with col2:
    st.caption(f"Mutação: {optimizer.INITIAL_MUTATION_PROBABILITY} | Restrições: {len(optimizer.ga.restriction_manager.get_active_restrictions())}")
with col3:
    restrictions_summary = []
    for restriction in optimizer.ga.restriction_manager.restrictions:
        if restriction.name == "fuel_restriction":
            restrictions_summary.append(f"Combustível: {restriction.max_distance}km")
        elif restriction.name == "vehicle_capacity_restriction":
            restrictions_summary.append(f"Capacidade: {restriction.max_patients_per_vehicle}")
    if restrictions_summary:
        st.caption(" | ".join(restrictions_summary[:2]))
 
# Main layout
st.subheader("🗺️ Visualização da Rota")
map_placeholder = st.empty()
 
st.subheader("📈 Evolução do Fitness")
fitness_placeholder = st.empty()
 
def create_map_mapbox(best_solution, population):
    """Cria mapa com Mapbox (para hospitais SP)"""
    fig = go.Figure()
 
    if best_solution:
        route_lats = [lat for lat, lon in best_solution] + [best_solution[0][0]]
        route_lons = [lon for lat, lon in best_solution] + [best_solution[0][1]]
 
        fig.add_trace(go.Scattermapbox(
            lat=route_lats,
            lon=route_lons,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Melhor Rota',
            hoverinfo='skip'
        ))
 
        if len(population) > 20:
            second_best = population[20]
            second_lats = [lat for lat, lon in second_best] + [second_best[0][0]]
            second_lons = [lon for lat, lon in second_best] + [second_best[0][1]]
 
            fig.add_trace(go.Scattermapbox(
                lat=second_lats,
                lon=second_lons,
                mode='lines',
                line=dict(width=1.5, color='lightgray'),
                name='2ª Melhor Rota',
                hoverinfo='skip'
            ))
 
    lats = [lat for lat, lon in optimizer.cities_locations]
    lons = [lon for lat, lon in optimizer.cities_locations]
 
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(size=10, color='red'),
        text=optimizer.city_names,
        hoverinfo='text',
        name='Hospitais'
    ))
 
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=-23.5505, lon=-46.6333),
            zoom=10.3
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest'
    )
 
    return fig
 
def create_map_pixels(best_solution, population):
    """Cria mapa com coordenadas em pixels (para ATT48)"""
    fig = go.Figure()
 
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
 
# Display map
current_best = optimizer.best_solutions[-1] if optimizer.best_solutions else None
current_pop = st.session_state.population if st.session_state.population else []
 
if optimizer.use_mapbox:
    fig = create_map_mapbox(current_best, current_pop)
else:
    fig = create_map_pixels(current_best, current_pop)
 
map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{st.session_state.generation}")
 
if optimizer.best_fitness_values:
    fitness_placeholder.line_chart(optimizer.best_fitness_values, height=300)
 
# Execute next generation
if next_gen and st.session_state.generation < optimizer.GENERATION_LIMIT:
    generation = next(optimizer.generation_counter)
    st.session_state.generation = generation

    # Prepara dados do veículo para integração das restrições (se múltiplos veículos estiver habilitado)
    vehicle_data_list = None
    multiple_vehicles_config = optimizer.config.get("restrictions.multiple_vehicles", {})
    if multiple_vehicles_config.get("enabled", False):
        multiple_vehicles_restriction = optimizer.ga.restriction_manager.get_restriction("multiple_vehicles_restriction")
        if multiple_vehicles_restriction:
            try:
                # Calcula dados por indivíduo (vehicles_used/unserved dependem da rota)
                vehicle_data_list = [
                    multiple_vehicles_restriction.get_vehicle_data_for_capacity_restriction(individual)
                    for individual in st.session_state.population
                ]
            except Exception as e:
                print(f"Aviso: Erro ao calcular vehicle_data_list: {e}")
                vehicle_data_list = None

    st.session_state.population, population_fitness = optimizer.evaluate_population(st.session_state.population, vehicle_data_list)
    best_fitness = population_fitness[0]
    best_solution = st.session_state.population[0]

    current_diversity = population_edge_diversity(st.session_state.population)
    stats = optimizer.ga.get_population_statistics(st.session_state.population, vehicle_data_list)
 
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

        # Prepara dados do veículo para integração das restrições (se múltiplos veículos estiver habilitado)
        vehicle_data_list = None
        multiple_vehicles_config = optimizer.config.get("restrictions.multiple_vehicles", {})
        if multiple_vehicles_config.get("enabled", False):
            multiple_vehicles_restriction = optimizer.ga.restriction_manager.get_restriction("multiple_vehicles_restriction")
            if multiple_vehicles_restriction:
                try:
                    # Calcula dados por indivíduo (vehicles_used/unserved dependem da rota)
                    vehicle_data_list = [
                        multiple_vehicles_restriction.get_vehicle_data_for_capacity_restriction(individual)
                        for individual in st.session_state.population
                    ]
                except Exception as e:
                    print(f"Aviso: Erro ao calcular vehicle_data_list: {e}")
                    vehicle_data_list = None

        st.session_state.population, population_fitness = optimizer.evaluate_population(st.session_state.population, vehicle_data_list)
        best_fitness = population_fitness[0]
        best_solution = st.session_state.population[0]

        current_diversity = population_edge_diversity(st.session_state.population)
        stats = optimizer.ga.get_population_statistics(st.session_state.population, vehicle_data_list)
 
        optimizer.track_progress(best_fitness, best_solution)
        optimizer.update_exploration_phase(generation)
        st.session_state.population = optimizer.manage_diversity(st.session_state.population, current_diversity, generation)
 
        status_text.info(f"🔄 Geração {generation}/{optimizer.GENERATION_LIMIT} | Melhor Fitness: {round(best_fitness, 2)}")
        progress_bar.progress(generation / optimizer.GENERATION_LIMIT)
 
        # Update visualizations
        if optimizer.use_mapbox:
            fig = create_map_mapbox(best_solution, st.session_state.population)
        else:
            fig = create_map_pixels(best_solution, st.session_state.population)
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
    st.rerun()
 
# Relatórios section
if optimizer.best_solutions:
    st.divider()
    st.subheader("📊 Relatórios")
 
    # 3.1 Ver detalhes da rota atual
    with st.expander("🗺️ Ver detalhes da rota atual", expanded=False):
        last_route = optimizer.best_solutions[-1]
        for idx, point in enumerate(last_route, 1):
            if optimizer.use_mapbox:
                city_idx = optimizer.cities_locations.index(point)
                st.write(f"{idx}. {optimizer.city_names[city_idx]}: ({point[0]:.4f}, {point[1]:.4f})")
            else:
                st.write(f"{idx}. Posição: {point}")
 
    # 3.2 Relatório de Performance das Rotas
    with st.expander("📈 Relatório de Performance", expanded=False):
        routes_data = []
        for i, solution in enumerate(optimizer.best_solutions[-10:], 1):
            fitness = optimizer.ga.calculate_fitness_with_restrictions(solution)
            routes_data.append({
                'route_id': i,
                'distance': fitness,
                'time': fitness / 50,
                'efficiency': max(0, 100 - (fitness / optimizer.fitness_target_solution * 100 - 100)) if optimizer.fitness_target_solution else 0,
                'violations': []
            })
 
        total_distance = sum(r['distance'] for r in routes_data)
        avg_efficiency = sum(r['efficiency'] for r in routes_data) / len(routes_data) if routes_data else 0
 
        report = f"""=== RELATÓRIO DE PERFORMANCE DAS ROTAS MÉDICAS ===
 
RESUMO EXECUTIVO:
- Total de rotas executadas: {len(routes_data)}
- Distância total percorrida: {total_distance:.2f}
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
    with st.expander("📋 Instruções da Rota", expanded=False):
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
Distância: {best_fitness:.2f}
Locais: {len(best_solution)}
Rota válida: {route_info['is_valid']}
Restrições ativas: {route_info['restrictions']}""")
        else:
            st.code(f"""=== INSTRUÇÕES DA ROTA ===
Distância: {best_fitness:.2f}
Locais: {len(best_solution)}
Rota válida: {route_info['is_valid']}
Restrições ativas: {route_info['restrictions']}""")
 
# Final message when simulation is complete
if st.session_state.generation >= optimizer.GENERATION_LIMIT:
    st.success("✅ Simulação concluída!")
 
    # 3.4 Estatísticas Finais (only shown when complete)
    with st.expander("📊 Estatísticas Finais", expanded=True):
        if optimizer.best_solutions:
            final_stats = optimizer.ga.get_population_statistics([optimizer.best_solutions[-1]])
            st.code(f"Population Statistics: {final_stats}")
 
            best_fitness = min(optimizer.best_fitness_values) if optimizer.best_fitness_values else 0
            target_fitness = optimizer.fitness_target_solution if optimizer.fitness_target_solution else best_fitness
            improvement = ((target_fitness - best_fitness) / target_fitness * 100) if target_fitness > 0 else 0
 
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
    with st.expander("⚙️ Configuração Detalhada", expanded=False):
        config_info = f"""=== CONFIGURAÇÃO COMPLETA ===
Limite de gerações: {optimizer.GENERATION_LIMIT}
Tamanho da população: {optimizer.POPULATION_SIZE}
Display: {optimizer.WIDTH}x{optimizer.HEIGHT}
Probabilidade de mutação: {optimizer.INITIAL_MUTATION_PROBABILITY}
Restrições ativas: {optimizer.ga.restriction_manager.get_active_restrictions()}"""
 
        for restriction in optimizer.ga.restriction_manager.restrictions:
            if restriction.name == "fuel_restriction":
                fuel_restriction = restriction
                config_info += f"\nRestrição de combustível: max {fuel_restriction.max_distance}km, custo {fuel_restriction.fuel_cost_per_km}/km"
                config_info += f"\nLimite de custo de combustível: R$ {fuel_restriction.fuel_cost_limit}"
            elif restriction.name == "vehicle_capacity_restriction":
                capacity_restriction = restriction
                config_info += f"\nCapacidade do veículo: max {capacity_restriction.max_patients_per_vehicle} entregas"
            elif restriction.name == "route_cost_restriction":
                route_cost_restriction = restriction
                config_info += f"\nCusto da rota: configurado"
            elif restriction.name == "multiple_vehicles_restriction":
                multiple_vehicles_restriction = restriction
                config_info += f"\nMúltiplos veículos: max {multiple_vehicles_restriction.max_vehicles} veículos, capacidade {multiple_vehicles_restriction.vehicle_capacity} pacientes/veículo"
 
        config_info += "\n" + "=" * 50
 
        st.code(config_info)