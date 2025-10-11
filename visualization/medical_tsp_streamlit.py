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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from config.route_cost import route_costs_hospital_sp
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
 
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
        forbidden_routes_config = self.config.get("restrictions.forbidden_routes", {})
        one_way_routes_config = self.config.get("restrictions.one_way_routes", {})
 
        # Usa configurações do sidebar se disponíveis, senão usa config.json
        if hasattr(self, 'sidebar_config') and self.sidebar_config:
            # Sidebar tem prioridade total
            capacity_enabled = self.sidebar_config.get('capacity_enabled', False)
            fixed_start_enabled = self.sidebar_config.get('fixed_start_enabled', False)
            multiple_vehicles_enabled = self.sidebar_config.get('multiple_vehicles_enabled', False)
            forbidden_routes_enabled = self.sidebar_config.get('forbidden_routes_enabled', False)
            one_way_routes_enabled = self.sidebar_config.get('one_way_routes_enabled', False)
 
            fuel_max_distance = self.sidebar_config.get('fuel_max_distance', 250.0)
            fuel_cost_per_km = self.sidebar_config.get('fuel_cost_per_km', 0.8)
            fuel_cost_limit = self.sidebar_config.get('fuel_cost_limit', 300.0)
            max_patients = self.sidebar_config.get('max_capacity', 10)
            max_vehicles_count = self.sidebar_config.get('max_vehicles', 5)
            forbidden_routes_penalty = self.sidebar_config.get('forbidden_routes_penalty', 1000.0)
            one_way_routes_penalty = self.sidebar_config.get('one_way_routes_penalty', 1000.0)
 
            print(f"DEBUG - Using sidebar config:")
            print(f"  capacity_enabled: {capacity_enabled}")
            print(f"  fixed_start_enabled: {fixed_start_enabled}")
            print(f"  multiple_vehicles_enabled: {multiple_vehicles_enabled}")
            print(f"  forbidden_routes_enabled: {forbidden_routes_enabled}")
            print(f"  one_way_routes_enabled: {one_way_routes_enabled}")
        else:
            # Usa config.json apenas se não houver sidebar_config
            capacity_enabled = capacity_config.get("enabled", True)
            fixed_start_enabled = hospital_config.get("enabled", True)
            multiple_vehicles_enabled = multiple_vehicles_config.get("enabled", True)
            forbidden_routes_enabled = forbidden_routes_config.get("enabled", True)
            one_way_routes_enabled = one_way_routes_config.get("enabled", True)
 
            fuel_max_distance = fuel_config.get("max_distance", 250.0)
            fuel_cost_per_km = fuel_config.get("fuel_cost_per_km", 0.8)
            fuel_cost_limit = fuel_config.get("fuel_cost_limit", None)
            max_patients = capacity_config.get("max_capacity", 10)
            max_vehicles_count = multiple_vehicles_config.get("max_vehicles", 5)
            forbidden_routes_penalty = forbidden_routes_config.get("base_distance_penalty", 1000.0)
            one_way_routes_penalty = one_way_routes_config.get("base_distance_penalty", 1000.0)
 
            print(f"DEBUG - Using config.json (no sidebar)")
 
        # Cria restrições apenas se estiverem habilitadas
        if capacity_enabled:
            capacity_restriction = VehicleCapacityRestriction(
                max_patients_per_vehicle=max_patients
            )
            capacity_restriction.set_weight(capacity_config.get("weight", 1.0))
            self.ga.restriction_manager.add_restriction(capacity_restriction)
            print("  + Capacity restriction added")
 
        if fixed_start_enabled:
            from restrictions.fixed_start_restriction import FixedStartRestriction
            hospital_restriction = FixedStartRestriction()
 
            if hospital_config.get("hospital_is_first_city", True):
                hospital_restriction.set_hospital_location(self.cities_locations[0])
 
            hospital_restriction.set_weight(hospital_config.get("weight", 5.0))
            self.ga.restriction_manager.add_restriction(hospital_restriction)
            print("  + Fixed start restriction added")
 
        if multiple_vehicles_enabled:
            vehicle_capacity = max_patients if capacity_enabled else 1
 
            multiple_vehicles_restriction = MultipleVehiclesRestriction(
                max_vehicles=max_vehicles_count,
                depot=self.cities_locations[0] if fixed_start_enabled else None,
                vehicle_capacity=vehicle_capacity
            )
            multiple_vehicles_restriction.set_weight(multiple_vehicles_config.get("weight", 2.0))
            self.ga.restriction_manager.add_restriction(multiple_vehicles_restriction)
            print("  + Multiple vehicles restriction added")
            
        # Cria e adiciona a restrição de rotas proibidas
        if forbidden_routes_enabled:
            forbidden_routes_restriction = ForbiddenRoutes(
                base_distance_penalty=forbidden_routes_penalty
            )
            forbidden_routes_restriction.set_weight(forbidden_routes_config.get("weight", 1.0))
            
            # Carregar rotas proibidas da configuração
            forbidden_routes_list = forbidden_routes_config.get("routes", [])
            for route in forbidden_routes_list:
                from_idx = route.get("from")
                to_idx = route.get("to")
                if from_idx is not None and to_idx is not None and from_idx < len(self.cities_locations) and to_idx < len(self.cities_locations):
                    forbidden_routes_restriction.add_forbidden_route(
                        self.cities_locations[from_idx], 
                        self.cities_locations[to_idx]
                    )
                    
            self.ga.restriction_manager.add_restriction(forbidden_routes_restriction)
            print("  + Forbidden routes restriction added")
            
        # Cria e adiciona a restrição de rotas unidirecionais
        if one_way_routes_enabled:
            one_way_routes_restriction = OneWayRoutes(
                base_distance_penalty=one_way_routes_penalty
            )
            one_way_routes_restriction.set_weight(one_way_routes_config.get("weight", 1.0))
            
            # Carregar rotas unidirecionais da configuração
            one_way_routes_list = one_way_routes_config.get("routes", [])
            for route in one_way_routes_list:
                from_idx = route.get("from")
                to_idx = route.get("to")
                if from_idx is not None and to_idx is not None and from_idx < len(self.cities_locations) and to_idx < len(self.cities_locations):
                    one_way_routes_restriction.add_one_way_route(
                        self.cities_locations[from_idx], 
                        self.cities_locations[to_idx]
                    )
                    
            self.ga.restriction_manager.add_restriction(one_way_routes_restriction)
            print("  + One-way routes restriction added")

        print(f"\nActive Restrictions: {self.ga.restriction_manager.get_active_restrictions()}")
        print(f"Total restrictions added: {len(self.ga.restriction_manager.restrictions)}\n")
 
    def setup_llm(self):
        llm_config = self.config.get("llm", {})
        if llm_config.get("enabled", True) and os.getenv("OPENAI_API_KEY"):
            open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.llm = LLMIntegration(open_ai_client)
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
            use_geographic = dataset_choice == 'hospitals_sp'
            vehicle_data = vehicle_data_list[i] if vehicle_data_list else None
            fitness = self.ga.calculate_fitness_with_restrictions(individual, vehicle_data, use_geographic)
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

    # Vehicle Capacity Restriction
    capacity_enabled = st.checkbox("Capacidade do Veículo", value=False, help="Limita número de pacientes por veículo")
    max_patients = 10
    if capacity_enabled:
        with st.expander("⚙️ Configurar Capacidade"):
            max_patients = st.slider("Pacientes por Veículo", min_value=1, max_value=20, value=10, step=1)

    # Fixed Start Restriction
    fixed_start_enabled = st.checkbox("Início Fixo (Hospital)", value=False, help="Força rota começar no hospital")

    # Multiple Vehicles Restriction
    multiple_vehicles_enabled = st.checkbox("Múltiplos Veículos", value=False, help="Permite distribuir pacientes entre várias ambulâncias")
    max_vehicles = 5
    if multiple_vehicles_enabled:
        with st.expander("⚙️ Configurar Veículos"):
            max_vehicles = st.slider("Número Máximo de Veículos", min_value=1, max_value=5, value=5, step=1)
            
    # Forbidden Routes Restriction
    forbidden_routes_enabled = st.checkbox("Rotas Proibidas", value=False, help="Define rotas que não podem ser percorridas")
    forbidden_routes_penalty = 1000.0
    if forbidden_routes_enabled:
        with st.expander("⚙️ Configurar Rotas Proibidas"):
            forbidden_routes_penalty = st.number_input("Penalidade Base", min_value=100.0, max_value=5000.0, value=1000.0, step=100.0)
            st.info("As rotas proibidas são definidas no arquivo de configuração. Para adicionar ou remover rotas proibidas específicas, edite o arquivo config/medical_tsp_config.json")
    
    # One-Way Routes Restriction
    one_way_routes_enabled = st.checkbox("Rotas Unidirecionais", value=False, help="Define rotas que só podem ser percorridas em uma direção")
    one_way_routes_penalty = 1000.0
    if one_way_routes_enabled:
        with st.expander("⚙️ Configurar Rotas Unidirecionais"):
            one_way_routes_penalty = st.number_input("Penalidade Base (Mão Única)", min_value=100.0, max_value=5000.0, value=1000.0, step=100.0)
            st.info("As rotas unidirecionais são definidas no arquivo de configuração. Para adicionar ou remover rotas unidirecionais específicas, edite o arquivo config/medical_tsp_config.json")
 
    st.divider()
 
    # Botão para aplicar configurações
    apply_config = st.button("🔄 Aplicar Configurações", use_container_width=True, type="primary")
 
    st.divider()
 
    # Summary of active restrictions
    active_count = sum([capacity_enabled, fixed_start_enabled, multiple_vehicles_enabled, forbidden_routes_enabled, one_way_routes_enabled])
    st.caption(f"✅ {active_count} restrições ativas")
 
    if dataset_choice == 'att48':
        st.caption("**ATT48 Benchmark**: 48 cidades em pixels com solução ótima conhecida")
    else:
        st.caption("**Hospitais SP**: 48 hospitais reais com mapa interativo")
 
# Initialize session state
if 'generation' not in st.session_state or st.session_state.get('dataset_type') != dataset_choice or apply_config:
    # Criar dicionário de configurações do sidebar
    sidebar_config = {}
    if capacity_enabled:
        sidebar_config.update({
            'capacity_enabled': capacity_enabled,
            'max_capacity': max_patients if capacity_enabled else 10,
        })
    else:
        sidebar_config['capacity_enabled'] = False

    sidebar_config['fixed_start_enabled'] = fixed_start_enabled

    if multiple_vehicles_enabled:
        sidebar_config.update({
            'multiple_vehicles_enabled': multiple_vehicles_enabled,
            'max_vehicles': max_vehicles if multiple_vehicles_enabled else 5,
        })
    else:
        sidebar_config['multiple_vehicles_enabled'] = False
        
    if forbidden_routes_enabled:
        sidebar_config.update({
            'forbidden_routes_enabled': forbidden_routes_enabled,
            'forbidden_routes_penalty': forbidden_routes_penalty if forbidden_routes_enabled else 1000.0,
        })
    else:
        sidebar_config['forbidden_routes_enabled'] = False
        
    if one_way_routes_enabled:
        sidebar_config.update({
            'one_way_routes_enabled': one_way_routes_enabled,
            'one_way_routes_penalty': one_way_routes_penalty if one_way_routes_enabled else 1000.0,
        })
    else:
        sidebar_config['one_way_routes_enabled'] = False
 
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
 
is_run_btn_disabled = st.session_state.generation == optimizer.GENERATION_LIMIT and st.session_state.generation > 0

# Control buttons
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    next_gen = st.button("▶️ Próxima Geração", use_container_width=True, type="primary", disabled=is_run_btn_disabled)
with col2:
    run_all = st.button("⏩ Executar Todas", use_container_width=True, disabled=is_run_btn_disabled)
with col3:
    reset_simulation = st.button("🔄 Resetar Simulação", use_container_width=True)
 
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

st.text("")
st.divider()
st.subheader("📈 Evolução do Fitness")
fitness_placeholder = st.empty()
 
@st.cache_data(ttl=60)
def create_map_mapbox(best_solution, population, cities_locations, city_names=None, forbidden_routes=None, one_way_routes=None, fixed_start_disabled=False, vehicle_routes=None):
    """Cria mapa com Mapbox (para hospitais SP)"""
    fig = go.Figure()

    # Desenha rotas proibidas se existirem
    if forbidden_routes:
        for route in forbidden_routes:
            city1, city2 = route
            # Encontrar nomes dos pontos ou usar coordenadas
            if city_names:
                try:
                    idx1 = [c for c in cities_locations].index(city1)
                    idx2 = [c for c in cities_locations].index(city2)
                    city1_name = city_names[idx1]
                    city2_name = city_names[idx2]
                except (ValueError, IndexError):
                    city1_name = f"({city1[0]:.4f}, {city1[1]:.4f})"
                    city2_name = f"({city2[0]:.4f}, {city2[1]:.4f})"
            else:
                city1_name = f"({city1[0]:.4f}, {city1[1]:.4f})"
                city2_name = f"({city2[0]:.4f}, {city2[1]:.4f})"
                
            route_name = f"Rota Proibida [{city1_name} → {city2_name}]"
            hover_text = f"Proibido: {city1_name} → {city2_name}"
            
            fig.add_trace(go.Scattermap(
                lat=[city1[0], city2[0]],
                lon=[city1[1], city2[1]],
                mode='lines',
                line=dict(width=3, color='red'),
                name=route_name,
                hoverinfo='text',
                hovertext=hover_text
            ))
    
    # Desenha rotas unidirecionais se existirem
    if one_way_routes:
        for route in one_way_routes:
            origin, destination = route
            # Calcula ponto médio para posicionar a seta
            mid_lat = (origin[0] + destination[0]) / 2
            mid_lon = (origin[1] + destination[1]) / 2
            
            # Calcula a direção da seta (vetor normalizado)
            dx = destination[0] - origin[0]
            dy = destination[1] - origin[1]
            dist = ((dx**2) + (dy**2))**0.5
            if dist > 0:
                dx, dy = dx/dist, dy/dist
            
            # Adiciona a linha da rota
            # Encontrar nomes dos pontos ou usar coordenadas
            if city_names:
                try:
                    idx1 = [c for c in cities_locations].index(origin)
                    idx2 = [c for c in cities_locations].index(destination)
                    origin_name = city_names[idx1]
                    dest_name = city_names[idx2]
                except (ValueError, IndexError):
                    origin_name = f"({origin[0]:.4f}, {origin[1]:.4f})"
                    dest_name = f"({destination[0]:.4f}, {destination[1]:.4f})"
            else:
                origin_name = f"({origin[0]:.4f}, {origin[1]:.4f})"
                dest_name = f"({destination[0]:.4f}, {destination[1]:.4f})"
                
            route_name = f"Mão Única [{origin_name} → {dest_name}]"
            hover_text = f"Sentido permitido: {origin_name} → {dest_name}"
            
            fig.add_trace(go.Scattermap(
                lat=[origin[0], destination[0]],
                lon=[origin[1], destination[1]],
                mode='lines',
                line=dict(width=2, color='green'),
                name=route_name,
                hoverinfo='text',
                hovertext=hover_text
            ))
            
            # Adiciona uma seta no meio da linha
            # Para Mapbox, usamos uma abordagem diferente com um marcador de seta
            # Cria um ponto um pouco adiante na linha para direcionar a seta
            arrow_lat = mid_lat + (dx * 0.001)  # Pequeno deslocamento na direção da linha
            arrow_lon = mid_lon + (dy * 0.001)
            
            fig.add_trace(go.Scattermap(
                lat=[mid_lat, arrow_lat],
                lon=[mid_lon, arrow_lon],
                mode='lines',
                line=dict(width=3, color='green'),
                marker=dict(size=10, symbol='arrow', angle=90 if dy > 0 else 270),
                name='Direção',
                hoverinfo='skip',
                showlegend=False
            ))

    if best_solution:
        # Cores para diferentes veículos
        vehicle_colors = ['blue', 'orange', 'purple', 'cyan', 'yellow', 'brown', 'pink']
        
        # Se temos rotas de veículos, desenhamos cada uma separadamente
        if vehicle_routes and len(vehicle_routes) > 0:
            for vehicle_id, route in vehicle_routes.items():
                # Seleciona uma cor para este veículo
                color_idx = vehicle_id % len(vehicle_colors)
                vehicle_color = vehicle_colors[color_idx]
                
                # Extrai coordenadas lat e lon para esta rota
                route_lats = [lat for lat, lon in route]
                route_lons = [lon for lat, lon in route]
                
                # Cria textos para hover com as coordenadas e informação do veículo
                route_hover_texts = []
                for i, point in enumerate(route):
                    point_lat, point_lon = point
                    
                    # Tenta encontrar o nome do hospital/local
                    point_name = "Local desconhecido"
                    if city_names:
                        try:
                            idx = [c for c in cities_locations].index(point)
                            point_name = city_names[idx]
                        except (ValueError, IndexError):
                            point_name = f"({point_lat:.4f}, {point_lon:.4f})"
                    
                    if i == 0 or i == len(route) - 1:  # Primeiro ou último ponto (depósito)
                        route_hover_texts.append(f"Depósito/Hospital: {point_name} - Veículo {vehicle_id+1}")
                    else:
                        route_hover_texts.append(f"{point_name} - Veículo {vehicle_id+1}")
                
                # Adiciona a rota deste veículo ao gráfico
                fig.add_trace(go.Scattermap(
                    lat=route_lats,
                    lon=route_lons,
                    mode='lines+markers',
                    line=dict(width=3, color=vehicle_color),
                    marker=dict(size=8, color=vehicle_color),
                    name=f'Veículo {vehicle_id+1}',
                    hoverinfo='text',
                    hovertext=route_hover_texts
                ))
        else:
            # Caso não tenhamos informações de veículos, mostramos a rota completa como antes
            route_lats = [lat for lat, lon in best_solution] + [best_solution[0][0]]
            route_lons = [lon for lat, lon in best_solution] + [best_solution[0][1]]
            
            # Cria textos para hover com informações dos locais
            route_hover_texts = []
            for i, point in enumerate(best_solution):
                point_lat, point_lon = point
                
                # Tenta encontrar o nome do hospital/local
                if city_names:
                    try:
                        idx = [c for c in cities_locations].index(point)
                        point_name = city_names[idx]
                        route_hover_texts.append(f"{point_name}: ({point_lat:.4f}, {point_lon:.4f})")
                    except (ValueError, IndexError):
                        route_hover_texts.append(f"Local {i+1}: ({point_lat:.4f}, {point_lon:.4f})")
                else:
                    route_hover_texts.append(f"Local {i+1}: ({point_lat:.4f}, {point_lon:.4f})")
                    
            # Adiciona o primeiro ponto novamente para fechar o ciclo
            if route_hover_texts:
                route_hover_texts.append(route_hover_texts[0])

            fig.add_trace(go.Scattermap(
                lat=route_lats,
                lon=route_lons,
                mode='lines+markers',
                line=dict(width=3, color='blue'),
                marker=dict(size=8, color='blue'),
                name='Melhor Rota',
                hoverinfo='text',
                hovertext=route_hover_texts
            ))

    lats = [lat for lat, lon in cities_locations]
    lons = [lon for lat, lon in cities_locations]
    
    # Adiciona todos os pontos com cor vermelha
    fig.add_trace(go.Scattermap(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(size=12, color='red'),
        text=city_names if city_names else [f"Local {i+1}" for i in range(len(cities_locations))],
        hoverinfo='text',
        name='Hospitais'
    ))
    
    # Destaca o ponto inicial com um marcador roxo maior
    # Se tiver uma rota e início fixo desativado, usa o primeiro ponto da rota
    # Caso contrário, usa o primeiro ponto da lista de cidades
    if best_solution and fixed_start_disabled:
        initial_point = best_solution[0]
        fig.add_trace(go.Scattermap(
            lat=[initial_point[0]],
            lon=[initial_point[1]],
            mode='markers',
            marker=dict(
                size=18, 
                color='purple',
                opacity=0.5
            ),
            name='Ponto Inicial',
            hoverinfo='text',
            hovertext='Ponto Inicial (Rota)'
        ))
    elif len(cities_locations) > 0:
        fig.add_trace(go.Scattermap(
            lat=[cities_locations[0][0]],
            lon=[cities_locations[0][1]],
            mode='markers',
            marker=dict(
                size=18, 
                color='purple',
                opacity=0.5
            ),
            name='Ponto Inicial',
            hoverinfo='text',
            hovertext='Ponto Inicial (Hospital)'
        ))

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=-23.5505, lon=-46.6333),
            zoom=8.9
        ),
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        hovermode='closest',
        # Reduzir animações para evitar piscadas
        transition_duration=300
    )

    return fig
 
@st.cache_data(ttl=60)
def create_map_pixels(best_solution, population, cities_locations, forbidden_routes=None, one_way_routes=None, fixed_start_disabled=False, vehicle_routes=None):
    """Cria mapa com coordenadas em pixels (para ATT48)"""
    fig = go.Figure()
    
    # Desenha rotas proibidas se existirem
    if forbidden_routes:
        for route in forbidden_routes:
            city1, city2 = route
            # Para o mapa de pixels, usamos índices como nomes
            try:
                city1_idx = cities_locations.index(city1)
                city2_idx = cities_locations.index(city2)
                city1_name = f"Cidade {city1_idx + 1}"
                city2_name = f"Cidade {city2_idx + 1}"
            except ValueError:
                city1_name = f"({city1[0]}, {city1[1]})"
                city2_name = f"({city2[0]}, {city2[1]})"
                
            route_name = f"Rota Proibida [{city1_name} → {city2_name}]"
            hover_text = f"Proibido: {city1_name} → {city2_name}"
            
            fig.add_trace(go.Scatter(
                x=[city1[0], city2[0]],
                y=[city1[1], city2[1]],
                mode='lines',
                line=dict(width=3, color='red'),
                name=route_name,
                hoverinfo='text',
                hovertext=hover_text
            ))
    
    # Desenha rotas unidirecionais se existirem
    if one_way_routes:
        for route in one_way_routes:
            origin, destination = route
            # Calcula ponto médio para posicionar a seta
            mid_x = (origin[0] + destination[0]) / 2
            mid_y = (origin[1] + destination[1]) / 2
            
            # Calcula a direção da seta (vetor normalizado)
            dx = destination[0] - origin[0]
            dy = destination[1] - origin[1]
            dist = ((dx**2) + (dy**2))**0.5
            if dist > 0:
                dx, dy = dx/dist, dy/dist
            
            # Adiciona a linha da rota
            # Para o mapa de pixels, usamos índices como nomes
            try:
                origin_idx = cities_locations.index(origin)
                dest_idx = cities_locations.index(destination)
                origin_name = f"Cidade {origin_idx + 1}"
                dest_name = f"Cidade {dest_idx + 1}"
            except ValueError:
                origin_name = f"({origin[0]}, {origin[1]})"
                dest_name = f"({destination[0]}, {destination[1]})"
                
            route_name = f"Mão Única [{origin_name} → {dest_name}]"
            hover_text = f"Sentido permitido: {origin_name} → {dest_name}"
            
            fig.add_trace(go.Scatter(
                x=[origin[0], destination[0]],
                y=[origin[1], destination[1]],
                mode='lines',
                line=dict(width=2, color='green'),
                name=route_name,
                hoverinfo='text',
                hovertext=hover_text
            ))
            
            # Adiciona uma seta no meio da linha
            arrow_angle = np.degrees(np.arctan2(dy, dx))
            fig.add_annotation(
                x=mid_x,
                y=mid_y,
                ax=mid_x + (dx * 20),  # Aponta na direção do fluxo
                ay=mid_y + (dy * 20),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='green',
                standoff=0
            )

    if best_solution:
        # Cores para diferentes veículos
        vehicle_colors = ['blue', 'orange', 'purple', 'cyan', 'yellow', 'brown', 'pink']
        
        # Se temos rotas de veículos, desenhamos cada uma separadamente
        if vehicle_routes and len(vehicle_routes) > 0:
            for vehicle_id, route in vehicle_routes.items():
                # Seleciona uma cor para este veículo
                color_idx = vehicle_id % len(vehicle_colors)
                vehicle_color = vehicle_colors[color_idx]
                
                # Extrai coordenadas x e y para esta rota
                route_x = [x for x, y in route]
                route_y = [y for x, y in route]
                
                # Cria textos para hover com as coordenadas e informação do veículo
                route_hover_texts = []
                for i, point in enumerate(route):
                    try:
                        city_idx = cities_locations.index(point)
                        if i == 0 or i == len(route) - 1:  # Primeiro ou último ponto (depósito)
                            route_hover_texts.append(f"Depósito/Hospital: ({point[0]}, {point[1]}) - Veículo {vehicle_id+1}")
                        else:
                            route_hover_texts.append(f"Cidade {city_idx+1}: ({point[0]}, {point[1]}) - Veículo {vehicle_id+1}")
                    except ValueError:
                        if i == 0 or i == len(route) - 1:  # Primeiro ou último ponto (depósito)
                            route_hover_texts.append(f"Depósito/Hospital: ({point[0]}, {point[1]}) - Veículo {vehicle_id+1}")
                        else:
                            route_hover_texts.append(f"Ponto {i}: ({point[0]}, {point[1]}) - Veículo {vehicle_id+1}")
                
                # Adiciona a rota deste veículo ao gráfico
                fig.add_trace(go.Scatter(
                    x=route_x,
                    y=route_y,
                    mode='lines+markers',
                    line=dict(width=3, color=vehicle_color),
                    marker=dict(size=8, color=vehicle_color),
                    name=f'Veículo {vehicle_id+1}',
                    hoverinfo='text',
                    hovertext=route_hover_texts
                ))
        else:
            # Caso não tenhamos informações de veículos, mostramos a rota completa como antes
            route_x = [x for x, y in best_solution] + [best_solution[0][0]]
            route_y = [y for x, y in best_solution] + [best_solution[0][1]]
            
            # Cria textos para hover com as coordenadas para a rota
            route_hover_texts = []
            for i, point in enumerate(best_solution):
                try:
                    city_idx = cities_locations.index(point)
                    route_hover_texts.append(f"Cidade {city_idx+1}: ({point[0]}, {point[1]})")
                except ValueError:
                    route_hover_texts.append(f"Ponto {i+1}: ({point[0]}, {point[1]})")
            # Adiciona o primeiro ponto novamente para fechar o ciclo
            if route_hover_texts:
                route_hover_texts.append(route_hover_texts[0])

            fig.add_trace(go.Scatter(
                x=route_x,
                y=route_y,
                mode='lines+markers',
                line=dict(width=3, color='blue'),
                marker=dict(size=8, color='red'),
                name='Melhor Rota',
                hoverinfo='text',
                hovertext=route_hover_texts
            ))

    cities_x = [x for x, y in cities_locations]
    cities_y = [y for x, y in cities_locations]
    
    # Adiciona todos os pontos com cor vermelha e texto com o índice
    # Cria rótulos para cada cidade com seu índice
    city_labels = [f"Cidade {i+1}" for i in range(len(cities_locations))]
    
    # Cria textos para hover com as coordenadas
    hover_texts = [f"Cidade {i+1}: ({x}, {y})" for i, (x, y) in enumerate(cities_locations)]
    
    # Adiciona os pontos com seus rótulos
    fig.add_trace(go.Scatter(
        x=cities_x,
        y=cities_y,
        mode='markers+text',
        marker=dict(size=12, color='red'),
        text=city_labels,
        textposition="top center",
        textfont=dict(size=8, color='black'),  # Fonte menor e cor preta para melhor visibilidade
        name='Cidades',
        hoverinfo='text',
        hovertext=hover_texts
    ))
    
    # Destaca o ponto inicial com um marcador roxo maior
    # Se tiver uma rota e início fixo desativado, usa o primeiro ponto da rota
    # Caso contrário, usa o primeiro ponto da lista de cidades
    if best_solution and fixed_start_disabled:
        initial_point = best_solution[0]
        # Cria texto para hover com as coordenadas do ponto inicial
        try:
            city_idx = cities_locations.index(initial_point)
            hover_text = f"Ponto Inicial (Rota) - Cidade {city_idx+1}: ({initial_point[0]}, {initial_point[1]})"
        except ValueError:
            hover_text = f"Ponto Inicial (Rota): ({initial_point[0]}, {initial_point[1]})"
            
        fig.add_trace(go.Scatter(
            x=[initial_point[0]],
            y=[initial_point[1]],
            mode='markers',
            marker=dict(
                size=18, 
                color='purple',
                opacity=0.5,
                line=dict(width=2, color='white')
            ),
            name='Ponto Inicial',
            hoverinfo='text',
            hovertext=hover_text
        ))
    elif len(cities_locations) > 0:
        # Para o mapa de pixels, podemos usar a propriedade line
        initial_point = cities_locations[0]
        hover_text = f"Ponto Inicial (Hospital) - Cidade 1: ({initial_point[0]}, {initial_point[1]})"
        
        fig.add_trace(go.Scatter(
            x=[initial_point[0]],
            y=[initial_point[1]],
            mode='markers',
            marker=dict(
                size=18, 
                color='purple',
                opacity=0.5,
                line=dict(width=2, color='white')
            ),
            name='Ponto Inicial',
            hoverinfo='text',
            hovertext=hover_text
        ))

    fig.update_layout(
        showlegend=True,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="X (pixels)",
        yaxis_title="Y (pixels)",
        # Reduzir animações para evitar piscadas
        transition_duration=300
    )
    
    # Configurações adicionais para melhorar a visualização dos textos
    fig.update_traces(
        selector=dict(mode='markers+text'),
        texttemplate='%{text}'
    )
    
    # Ajusta o layout para evitar sobreposição de textos
    fig.update_layout(
        annotations=[],  # Remove anotações automáticas que podem causar sobreposição
        plot_bgcolor='rgba(240, 240, 240, 0.5)',  # Fundo mais claro para melhor contraste
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.2)'
        )
    )

    return fig
 
# Display map
current_best = optimizer.best_solutions[-1] if optimizer.best_solutions else None
current_pop = st.session_state.population if st.session_state.population else []

# Obter rotas proibidas, unidirecionais e informações de veículos se as restrições estiverem ativas
forbidden_routes = None
one_way_routes = None
vehicle_routes = None
fixed_start_disabled = True  # Por padrão, considera desativado

# Verificar se as restrições estão ativas e obter as rotas
for restriction in optimizer.ga.restriction_manager.restrictions:
    if restriction.name == "forbidden_routes":
        forbidden_routes = list(restriction.get_all_forbidden_routes())
    elif restriction.name == "one_way_routes":
        one_way_routes = list(restriction.get_all_one_way_routes())
    elif restriction.name == "fixed_start_restriction":
        # Se encontrou a restrição de início fixo, ela está ativa
        fixed_start_disabled = False
    elif restriction.name == "multiple_vehicles_restriction" and current_best:
        # Se a restrição de múltiplos veículos estiver ativa e tivermos uma solução,
        # obtém as rotas por veículo
        multiple_vehicles_restriction = restriction
        
        # Remove o depósito da rota para contar apenas os pacientes
        depot = multiple_vehicles_restriction.depot
        if depot:
            patients = [city for city in current_best if city != depot]
            
            # Distribui pacientes entre veículos e obtém as rotas
            vehicle_routes = multiple_vehicles_restriction._distribute_patients_to_vehicles(
                patients, 
                depot, 
                multiple_vehicles_restriction.vehicle_capacity,
                multiple_vehicles_restriction.max_vehicles
            )
            print(f"DEBUG: Obtidas {len(vehicle_routes)} rotas de veículos")

if optimizer.use_mapbox:
    fig = create_map_mapbox(
        current_best, 
        current_pop, 
        optimizer.cities_locations, 
        optimizer.city_names,
        forbidden_routes,
        one_way_routes,
        fixed_start_disabled,
        vehicle_routes
    )
else:
    fig = create_map_pixels(
        current_best, 
        current_pop, 
        optimizer.cities_locations,
        forbidden_routes,
        one_way_routes,
        fixed_start_disabled,
        vehicle_routes
    )

# Usar um key estático para evitar recriação desnecessária
map_placeholder.plotly_chart(fig, use_container_width=True, key="initial_map_view")
 
if optimizer.best_fitness_values:
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the fitness values
    generations = range(1, len(optimizer.best_fitness_values) + 1)
    ax.plot(generations, optimizer.best_fitness_values, 'b-', linewidth=2)
    
    # Customize the plot
    ax.set_title("Evolução do Fitness", fontsize=14, fontweight='bold')
    ax.set_xlabel("Geração", fontsize=12)
    ax.set_ylabel("Fitness (Distância)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Display in Streamlit
    fitness_placeholder.pyplot(fig)
    plt.close(fig)
 
# Execute next generation
if next_gen and st.session_state.generation < optimizer.GENERATION_LIMIT:
    # Executar várias gerações de uma vez para reduzir atualizações da UI
    num_generations_per_click = 5
    for _ in range(num_generations_per_click):
        if st.session_state.generation >= optimizer.GENERATION_LIMIT:
            break

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
 
        # Update visualizations apenas a cada 5 gerações para reduzir piscadas
        if generation % 5 == 0:
            # Obter rotas proibidas, unidirecionais e informações de veículos se as restrições estiverem ativas
            forbidden_routes = None
            one_way_routes = None
            vehicle_routes = None
            fixed_start_disabled = True  # Por padrão, considera desativado

            # Verificar se as restrições estão ativas e obter as rotas
            for restriction in optimizer.ga.restriction_manager.restrictions:
                if restriction.name == "forbidden_routes":
                    forbidden_routes = list(restriction.get_all_forbidden_routes())
                elif restriction.name == "one_way_routes":
                    one_way_routes = list(restriction.get_all_one_way_routes())
                elif restriction.name == "fixed_start_restriction":
                    # Se encontrou a restrição de início fixo, ela está ativa
                    fixed_start_disabled = False
                elif restriction.name == "multiple_vehicles_restriction" and best_solution:
                    # Se a restrição de múltiplos veículos estiver ativa e tivermos uma solução,
                    # obtém as rotas por veículo
                    multiple_vehicles_restriction = restriction
                    
                    # Remove o depósito da rota para contar apenas os pacientes
                    depot = multiple_vehicles_restriction.depot
                    if depot:
                        patients = [city for city in best_solution if city != depot]
                        
                        # Distribui pacientes entre veículos e obtém as rotas
                        vehicle_routes = multiple_vehicles_restriction._distribute_patients_to_vehicles(
                            patients, 
                            depot, 
                            multiple_vehicles_restriction.vehicle_capacity,
                            multiple_vehicles_restriction.max_vehicles
                        )
                    
            if optimizer.use_mapbox:
                fig = create_map_mapbox(
                    best_solution, 
                    st.session_state.population,
                    optimizer.cities_locations,
                    optimizer.city_names,
                    forbidden_routes,
                    one_way_routes,
                    fixed_start_disabled,
                    vehicle_routes
                )
            else:
                fig = create_map_pixels(
                    best_solution, 
                    st.session_state.population,
                    optimizer.cities_locations,
                    forbidden_routes,
                    one_way_routes,
                    fixed_start_disabled,
                    vehicle_routes
                )
            # Usar uma chave única baseada na geração atual
            map_key = f"map_run_gen_{generation}"
            map_placeholder.plotly_chart(fig, use_container_width=True, key=map_key)

            if optimizer.best_fitness_values:
                # Create matplotlib figure
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # Plot the fitness values
                generations = range(1, len(optimizer.best_fitness_values) + 1)
                ax.plot(generations, optimizer.best_fitness_values, 'b-', linewidth=2)
                
                # Customize the plot
                ax.set_title("Evolução do Fitness", fontsize=14, fontweight='bold')
                ax.set_xlabel("Geração", fontsize=12)
                ax.set_ylabel("Fitness (Distância)", fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Adjust layout
                plt.tight_layout()
                
                # Display in Streamlit
                fitness_placeholder.pyplot(fig)
                plt.close(fig)
 
        st.session_state.population = optimizer.create_new_generation(st.session_state.population, population_fitness, current_diversity)
 
        optimizer.update_mutation_parameters()
        optimizer.print_generation_info(generation, best_fitness, current_diversity, stats)
 
        if generation >= optimizer.GENERATION_LIMIT:
            optimizer.final_report()
            break
 
        time.sleep(0.1)
 
    progress_bar.empty()
    st.rerun()

if reset_simulation:
    optimizer.generation_counter = itertools.count(start=1)
    optimizer.mutation_intensity = optimizer.INITIAL_MUTATION_INTENSITY
    optimizer.mutation_probability = optimizer.INITIAL_MUTATION_PROBABILITY
    optimizer.best_fitness_values = []
    optimizer.best_solutions = []
    optimizer.last_best_fitness = None
    optimizer.diversity_history = []
    optimizer.generation_without_improvement = 0
    optimizer.finished_exploration = False

    # Resetar session state
    st.session_state.generation = 0
    st.session_state.population = optimizer.create_initial_population()

    st.success("✅ Simulação resetada! População inicial criada.")
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

    multiple_vehicle_solution = {'vehicles_used': 1}
    best_solution = optimizer.best_solutions[-1]

    if multiple_vehicles_enabled:
        multiple_vehicle_solution = optimizer.ga.restriction_manager.get_restriction("multiple_vehicles_restriction").get_vehicle_data_for_capacity_restriction(best_solution)

    # 3.2 Relatório de Performance das Rotas
    with st.expander("📈 Relatório de Performance", expanded=False):
        routes_data = []
        for i, solution in enumerate(optimizer.best_solutions[-10:], 1):
            use_geographic = dataset_choice == 'hospitals_sp'
            fitness = optimizer.ga.calculate_fitness_with_restrictions(solution, use_geographic=use_geographic)
            routes_data.append({
                'route_id': i,
                'distance': fitness,
                'time': fitness / 50,
                'violations': [],
                'vehicles_used': multiple_vehicle_solution['vehicles_used']
            })
 
        total_distance = sum(r['distance'] for r in routes_data)
 
        report = f"""=== RELATÓRIO DE PERFORMANCE DAS ROTAS MÉDICAS ===
 
RESUMO EXECUTIVO:
- Total de rotas executadas: {len(routes_data)}
- Distância total percorrida: {total_distance:.2f}
- Quantidade de veículos: {multiple_vehicle_solution['vehicles_used']}
 
ANÁLISE DE PERFORMANCE:
O sistema de otimização está funcionando adequadamente. 
Recomenda-se monitoramento contínuo para identificar oportunidades de melhoria.
 
PRÓXIMOS PASSOS:
1. Configurar integração completa com LLM
2. Implementar alertas automáticos
3. Expandir coleta de dados de performance
 
OBSERVAÇÃO: Relatório em modo fallback. Configure LLM para análises detalhadas."""
 
        if optimizer.llm:
            report = optimizer.llm.generate_route_report(routes_data, "daily")
            st.markdown(report)
        else:
            st.markdown(report)
        st.download_button("Baixar relatório executivo", data=report, file_name="relatorio-performance.md")
 
    # 3.3 Instruções da Rota
    with st.expander("📋 Instruções da Rota", expanded=False):
        best_solution = optimizer.best_solutions[-1]
        best_fitness = optimizer.best_fitness_values[-1]

        route_info = {
            "distance": best_fitness,
            "restrictions": optimizer.ga.restriction_manager.get_active_restrictions(),
            "is_valid": optimizer.ga.restriction_manager.validate_route(best_solution),
            "vehicles_used": multiple_vehicle_solution['vehicles_used']
        }
 
        if optimizer.llm:
            try:
                instructions = optimizer.llm.generate_delivery_instructions(best_solution, route_info)
                st.markdown(instructions)
                st.download_button("Baixar instrução de rota", data=instructions, file_name="relatorio-rota.md")
            except:
                st.markdown(f"""=== INSTRUÇÕES DA ROTA ===
Distância: {best_fitness:.2f}
Locais: {len(best_solution)}
Rota válida: {route_info['is_valid']}
Quantidade de rotas: {multiple_vehicle_solution['vehicles_used']}
Restrições ativas: {route_info['restrictions']}""")
        else:
            st.markdown(f"""=== INSTRUÇÕES DA ROTA ===
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
 
            final_report = f"""=== RELATÓRIO FINAL DE OTIMIZAÇÃO ===
Total de gerações: {len(optimizer.best_fitness_values)}
Melhor fitness alcançado: {best_fitness:.2f}
Estatísticas da solução final: {final_stats}
Configuração utilizada:
- Veículos utilizados: {multiple_vehicle_solution}
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
            elif restriction.name == "forbidden_routes":
                forbidden_routes_restriction = restriction
                routes_count = len(forbidden_routes_restriction.get_all_forbidden_routes())
                config_info += f"\nRotas proibidas: {routes_count} rotas, penalidade {forbidden_routes_restriction._base_distance_penalty}"
            elif restriction.name == "one_way_routes":
                one_way_routes_restriction = restriction
                routes_count = len(one_way_routes_restriction.get_all_one_way_routes())
                config_info += f"\nRotas unidirecionais: {routes_count} rotas, penalidade {one_way_routes_restriction._base_distance_penalty}"
 
        config_info += "\n" + "=" * 50
 
        st.code(config_info)