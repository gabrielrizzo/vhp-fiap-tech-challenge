import streamlit as st
import plotly.graph_objects as go
from benchmark_hospitals_sp import hospitals_sp_data
from genetic_algorithm import order_crossover, generate_random_population, calculate_fitness, sort_population, generate_nearest_neightbor, mutate_hard
from selection_functions import tournament_or_rank_based_selection
import random
import time

# Extract locations and names
locations = [(d['lat'], d['lon']) for d in hospitals_sp_data]
names = [d['name'] for d in hospitals_sp_data]

# GA parameters
N_CITIES = 48
POPULATION_SIZE = 1000
N_GENERATIONS = 200
N_NEIGHTBORS = 50
INTIAL_MUTATION_INTENSITY = 40
INITIAL_MUTATION_PROBABILITY = 0.85
AFTER_EXPLORATION_MUTATION_INTENSITY = 30
AFTER_EXPLORATION_MUTATION_PROBABILITY = 0.5
N_EXPLORATION_GENERATION = 30

# Page config
st.set_page_config(page_title="TSP GA - São Paulo", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if 'generation' not in st.session_state:
    st.session_state.generation = 0
    st.session_state.population = generate_random_population(locations, POPULATION_SIZE - N_NEIGHTBORS)
    for _ in range(N_NEIGHTBORS):
        nearest_neighbor = generate_nearest_neightbor(locations, random.randint(0, N_CITIES - 1))
        st.session_state.population.append(nearest_neighbor)
    st.session_state.mutation_intensity = INTIAL_MUTATION_INTENSITY
    st.session_state.mutation_probability = INITIAL_MUTATION_PROBABILITY
    st.session_state.finished_exploration = False
    st.session_state.generation_same_fitness_counter = 0
    st.session_state.last_best_fitness = None
    st.session_state.best_fitness_values = []
    st.session_state.best_solutions = []

# Header
st.title("🧬 Algoritmo Genético - Problema do Caixeiro Viajante")
st.markdown("**Hospitais e Clínicas de São Paulo**")

# Info metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Geração Atual", f"{st.session_state.generation}/{N_GENERATIONS}")
with col2:
    if st.session_state.best_fitness_values:
        st.metric("Melhor Distância", f"{round(st.session_state.best_fitness_values[-1], 2)} km")
    else:
        st.metric("Melhor Distância", "---")
with col3:
    st.metric("População", POPULATION_SIZE)
with col4:
    fase = "Exploração" if not st.session_state.finished_exploration else "Refinamento"
    st.metric("Fase", fase)

st.divider()

# Control buttons
col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    next_gen = st.button("▶️ Próxima Geração", use_container_width=True, type="primary")
with col2:
    run_all = st.button("⏩ Executar Todas", use_container_width=True)

st.divider()

# Main layout: Map on left, Chart on right
col_map, col_chart = st.columns([2, 1])

with col_map:
    st.subheader("🗺️ Mapa de São Paulo")
    map_placeholder = st.empty()

with col_chart:
    st.subheader("📈 Evolução do Fitness")
    fitness_placeholder = st.empty()

def create_map(best_solution, generation, population):
    """Helper function to create the map figure"""
    fig = go.Figure()
    
    # Add best route first (so it appears behind markers)
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
        
        # Add second best route
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
    
    # Add markers on top
    lats = [lat for lat, lon in locations]
    lons = [lon for lat, lon in locations]
    
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode='markers',
        marker=dict(size=10, color='red'),
        text=names,
        hoverinfo='text',
        name='Hospitais'
    ))
    
    # Update layout with mapbox
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

# Display initial or current state
current_best = st.session_state.best_solutions[-1] if st.session_state.best_solutions else None
current_pop = st.session_state.population if st.session_state.population else []

fig = create_map(current_best, st.session_state.generation, current_pop)
map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_{st.session_state.generation}")

if st.session_state.best_fitness_values:
    fitness_placeholder.line_chart(st.session_state.best_fitness_values, height=400)

# Run all generations
if run_all and st.session_state.generation < N_GENERATIONS:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for gen in range(st.session_state.generation + 1, N_GENERATIONS + 1):
        st.session_state.generation = gen
        
        population_fitness = [calculate_fitness(individual) for individual in st.session_state.population]
        st.session_state.population, population_fitness = sort_population(st.session_state.population, population_fitness)
        
        best_fitness = calculate_fitness(st.session_state.population[0])
        best_solution = st.session_state.population[0]
        
        if st.session_state.finished_exploration:
            if st.session_state.last_best_fitness == best_fitness:
                st.session_state.generation_same_fitness_counter += 1
            else:
                st.session_state.generation_same_fitness_counter = 0
        
        st.session_state.last_best_fitness = best_fitness
        st.session_state.best_fitness_values.append(best_fitness)
        st.session_state.best_solutions.append(best_solution)
        
        status_text.info(f"🔄 Geração {gen}/{N_GENERATIONS} | Melhor Distância: {round(best_fitness, 2)} km")
        progress_bar.progress(gen / N_GENERATIONS)
        
        # Update map
        fig = create_map(best_solution, gen, st.session_state.population)
        map_placeholder.plotly_chart(fig, use_container_width=True, key=f"map_run_{gen}")
        
        # Update fitness chart
        fitness_placeholder.line_chart(st.session_state.best_fitness_values, height=400)
        
        new_population = [st.session_state.population[0]]  # Elitism
        
        if gen > N_EXPLORATION_GENERATION and not st.session_state.finished_exploration:
            st.session_state.mutation_intensity = AFTER_EXPLORATION_MUTATION_INTENSITY
            st.session_state.mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY
            st.session_state.finished_exploration = True
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = tournament_or_rank_based_selection(st.session_state.population, population_fitness, tournament_prob=0.7)
            child1, child2 = order_crossover(parent1, parent2)
            
            if st.session_state.generation_same_fitness_counter == 100:
                if st.session_state.mutation_intensity < N_CITIES:
                    st.session_state.mutation_intensity += 2
                if st.session_state.mutation_probability < 0.85:
                    st.session_state.mutation_probability += 0.05
                st.session_state.generation_same_fitness_counter = 0
            
            child1 = mutate_hard(child1, st.session_state.mutation_probability, intensity=st.session_state.mutation_intensity)
            child2 = mutate_hard(child2, st.session_state.mutation_probability, intensity=st.session_state.mutation_intensity)
            
            new_population.append(child1)
            new_population.append(child2)
        
        st.session_state.population = new_population
        time.sleep(0.2)
    
    progress_bar.empty()
    status_text.success(f"✅ Simulação concluída! Melhor distância: {round(st.session_state.best_fitness_values[-1], 2)} km")
    st.rerun()

# Execute next generation
elif next_gen and st.session_state.generation < N_GENERATIONS:
    st.session_state.generation += 1
    
    population_fitness = [calculate_fitness(individual) for individual in st.session_state.population]
    st.session_state.population, population_fitness = sort_population(st.session_state.population, population_fitness)
    
    best_fitness = calculate_fitness(st.session_state.population[0])
    best_solution = st.session_state.population[0]
    
    if st.session_state.finished_exploration:
        if st.session_state.last_best_fitness == best_fitness:
            st.session_state.generation_same_fitness_counter += 1
        else:
            st.session_state.generation_same_fitness_counter = 0
    
    st.session_state.last_best_fitness = best_fitness
    st.session_state.best_fitness_values.append(best_fitness)
    st.session_state.best_solutions.append(best_solution)
    
    new_population = [st.session_state.population[0]]  # Elitism
    
    if st.session_state.generation > N_EXPLORATION_GENERATION and not st.session_state.finished_exploration:
        st.info('🔍 Fase de exploração finalizada! Iniciando refinamento...')
        st.session_state.mutation_intensity = AFTER_EXPLORATION_MUTATION_INTENSITY
        st.session_state.mutation_probability = AFTER_EXPLORATION_MUTATION_PROBABILITY
        st.session_state.finished_exploration = True
    
    while len(new_population) < POPULATION_SIZE:
        parent1, parent2 = tournament_or_rank_based_selection(st.session_state.population, population_fitness, tournament_prob=0.7)
        child1, child2 = order_crossover(parent1, parent2)
        
        if st.session_state.generation_same_fitness_counter == 100:
            if st.session_state.mutation_intensity < N_CITIES:
                st.session_state.mutation_intensity += 2
            if st.session_state.mutation_probability < 0.85:
                st.session_state.mutation_probability += 0.05
            st.session_state.generation_same_fitness_counter = 0
        
        child1 = mutate_hard(child1, st.session_state.mutation_probability, intensity=st.session_state.mutation_intensity)
        child2 = mutate_hard(child2, st.session_state.mutation_probability, intensity=st.session_state.mutation_intensity)
        
        new_population.append(child1)
        new_population.append(child2)
    
    st.session_state.population = new_population
    st.rerun()

# Display route details
if st.session_state.best_solutions:
    st.divider()
    with st.expander("🗺️ Ver detalhes da rota atual", expanded=False):
        route_indices = []
        for point in st.session_state.best_solutions[-1]:
            for i, loc in enumerate(locations):
                if loc == point:
                    route_indices.append(i)
                    break
        route_names = [names[i] for i in route_indices]
        
        for idx, name in enumerate(route_names, 1):
            st.write(f"{idx}. {name}")

# Final message
if st.session_state.generation >= N_GENERATIONS:
    st.success("✅ Simulação concluída!")
    st.balloons()