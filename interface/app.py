import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algorithm import (
    generate_random_population,
    calculate_fitness,
    order_crossover,
    uniform_crossover,
    mutate,
    mutate_hard,
    sort_population,
)
from restrictions.forbidden_routes import ForbiddenRoutes

# Configuração da página
st.set_page_config(
    page_title="TSP - Rotas Médicas",
    page_icon="🚑",
    layout="wide"
)

# Título
st.title("🚑 Otimização de Rotas Médicas")
st.markdown("---")

# Sidebar para controles
with st.sidebar:
    st.header("Configurações")
    
    # Parâmetros do algoritmo genético
    st.subheader("Algoritmo Genético")
    n_cities = st.slider("Número de Cidades", 5, 50, 10)
    population_size = st.slider("Tamanho da População", 10, 200, 100)
    n_generations = st.slider("Número de Gerações", 10, 10000, 100)
    mutation_probability = st.slider("Probabilidade de Mutação", 0.0, 1.0, 0.3)
    
    # Seleção de métodos
    st.subheader("Métodos")
    crossover_method = st.selectbox(
        "Método de Crossover",
        ["Order Crossover", "Uniform Crossover"],
        index=0
    )
    mutation_method = st.selectbox(
        "Método de Mutação",
        ["Simple Mutation", "Hard Mutation"],
        index=0
    )
    if mutation_method == "Hard Mutation":
        mutation_intensity = st.slider("Intensidade da Mutação", 2, 10, 7)
    
    # Restrições
    st.subheader("Restrições")
    use_forbidden_routes = st.checkbox("Usar Rotas Proibidas", value=False)
    if use_forbidden_routes:
        n_forbidden_routes = st.slider(
            "Número de Rotas Proibidas",
            1,
            n_cities * (n_cities - 1) // 4,  # Máximo de 25% das rotas possíveis
            n_cities
        )

# Layout principal com duas colunas
col1, col2 = st.columns([2, 1])

# Coluna do mapa
with col1:
    st.subheader("Mapa de Rotas")
    
    # Função para criar o mapa
    def plot_route(
        cities: List[Tuple[float, float]],
        current_route: List[Tuple[float, float]] = None,
        best_route: List[Tuple[float, float]] = None,
        forbidden_routes: ForbiddenRoutes = None
    ):
        fig = go.Figure()
        
        # Plotar cidades
        x = [city[0] for city in cities]
        y = [city[1] for city in cities]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            name='Cidades',
            text=[f'Cidade {i}' for i in range(len(cities))],
            marker=dict(size=10, color='red'),
        ))
        
        # Plotar melhor rota (se existir)
        if best_route:
            x_route = [city[0] for city in best_route + [best_route[0]]]
            y_route = [city[1] for city in best_route + [best_route[0]]]
            fig.add_trace(go.Scatter(
                x=x_route,
                y=y_route,
                mode='lines',
                name='Melhor Rota',
                line=dict(color='green', width=2),
            ))
        
        # Plotar rota atual (se existir)
        if current_route:
            x_current = [city[0] for city in current_route + [current_route[0]]]
            y_current = [city[1] for city in current_route + [current_route[0]]]
            fig.add_trace(go.Scatter(
                x=x_current,
                y=y_current,
                mode='lines',
                name='Rota Atual',
                line=dict(color='blue', width=2, dash='dash'),
            ))
        
        # Plotar rotas proibidas (se existirem)
        if forbidden_routes:
            for route in forbidden_routes.get_all_forbidden_routes():
                city1, city2 = route
                fig.add_trace(go.Scatter(
                    x=[city1[0], city2[0]],
                    y=[city1[1], city2[1]],
                    mode='lines',
                    name='Rota Proibida',
                    line=dict(color='red', width=2),
                ))
        
        # Configurar layout
        fig.update_layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor='white',
            width=800,
            height=600
        )
        
        return fig

    # Placeholder para o mapa
    map_placeholder = st.empty()

# Coluna de informações
with col2:
    st.subheader("Informações")
    
    # Container para estatísticas
    stats_container = st.container()
    
    # Placeholders para informações
    with stats_container:
        generation_info = st.empty()
        fitness_info = st.empty()
        route_info = st.empty()
        
        # Gráfico de evolução do fitness
        st.subheader("Evolução do Fitness")
        fitness_chart = st.line_chart()
        
        # Histórico de melhorias
        st.subheader("Histórico de Melhorias")
        improvements_container = st.container()
    
    # Botões de controle
    col_start, col_clear = st.columns(2)
    with col_start:
        start_button = st.button("Iniciar Otimização")
    with col_clear:
        if st.button("Limpar Histórico"):
            st.session_state.fitness_history = []
            st.session_state.best_fitness_ever = float('inf')
            st.session_state.generation_of_best = 0
            st.rerun()

# Inicializar histórico de fitness se não existir
if 'fitness_history' not in st.session_state:
    st.session_state.fitness_history = []
    st.session_state.best_fitness_ever = float('inf')
    st.session_state.generation_of_best = 0

# Estado da aplicação
if 'cities' not in st.session_state:
    st.session_state.cities = [
        (np.random.randint(0, 100), np.random.randint(0, 100))
        for _ in range(n_cities)
    ]

if 'forbidden_routes' not in st.session_state:
    st.session_state.forbidden_routes = ForbiddenRoutes()

if 'best_route' not in st.session_state:
    st.session_state.best_route = None

if 'current_route' not in st.session_state:
    st.session_state.current_route = None

# Atualizar o mapa inicial
map_placeholder.plotly_chart(
    plot_route(
        st.session_state.cities,
        st.session_state.current_route,
        st.session_state.best_route,
        st.session_state.forbidden_routes if use_forbidden_routes else None
    )
)

# Lógica principal
if start_button:
    # Resetar rotas proibidas se necessário
    if use_forbidden_routes:
        forbidden_routes = ForbiddenRoutes()
        # Gerar rotas proibidas aleatórias
        for _ in range(n_forbidden_routes):
            # Selecionar duas cidades aleatórias
            indices = np.random.choice(len(st.session_state.cities), 2, replace=False)
            city1 = st.session_state.cities[indices[0]]
            city2 = st.session_state.cities[indices[1]]
            forbidden_routes.add_forbidden_route(city1, city2)
        st.session_state.forbidden_routes = forbidden_routes
    else:
        forbidden_routes = None
    
    # Criar população inicial
    population = generate_random_population(st.session_state.cities, population_size)
    
    # Loop principal
    for generation in range(n_generations):
        # Calcular fitness
        population_fitness = [
            calculate_fitness(individual, forbidden_routes)
            for individual in population
        ]
        
        # Ordenar população
        population, population_fitness = sort_population(population, population_fitness)
        
        # Atualizar melhor solução
        best_solution = population[0]
        best_fitness = population_fitness[0]
        
        # Atualizar estado
        st.session_state.best_route = best_solution
        st.session_state.current_route = population[-1]  # Mostra o pior indivíduo como rota atual
        
        # Atualizar histórico de fitness
        st.session_state.fitness_history.append(best_fitness)
        
        # Verificar se é a melhor solução já encontrada
        if best_fitness < st.session_state.best_fitness_ever:
            st.session_state.best_fitness_ever = best_fitness
            st.session_state.generation_of_best = generation + 1
            with improvements_container:
                st.success(f"Nova melhor solução encontrada na geração {generation + 1} com fitness {best_fitness:.2f}")
        
        # Atualizar interface
        generation_info.write(f"Geração Atual: {generation + 1}/{n_generations}")
        fitness_info.write(f"""
        Melhor Fitness Atual: {best_fitness:.2f}
        Melhor Fitness Global: {st.session_state.best_fitness_ever:.2f} (Geração {st.session_state.generation_of_best})
        """)
        route_info.write("Melhor Rota: " + " → ".join([f"Cidade {i}" for i in range(len(best_solution))]))
        
        # Atualizar gráfico de evolução
        fitness_chart.line_chart(st.session_state.fitness_history)
        
        # Atualizar mapa
        map_placeholder.plotly_chart(
            plot_route(
                st.session_state.cities,
                st.session_state.current_route,
                st.session_state.best_route,
                st.session_state.forbidden_routes if use_forbidden_routes else None
            )
        )
        
        # Criar nova população
        new_population = [population[0]]  # Elitismo
        
        while len(new_population) < population_size:
            # Seleção
            # Selecionar dois pais aleatórios dos 10 melhores indivíduos
            indices = np.random.choice(10, 2, replace=False)
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]
            
            # Crossover
            if crossover_method == "Order Crossover":
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = uniform_crossover(parent1, parent2)
            
            # Mutação
            if mutation_method == "Simple Mutation":
                child1 = mutate(child1, mutation_probability)
            else:
                child1 = mutate_hard(child1, mutation_probability, mutation_intensity)
            
            new_population.append(child1)
        
        population = new_population
        
        # Pequena pausa para visualização
        if generation % 5 == 0:
            st.rerun()
