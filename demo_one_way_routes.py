import random
from typing import List, Tuple
from genetic_algorithm import (
    generate_random_population,
    calculate_fitness,
    order_crossover,
    mutate,
    sort_population
)
from restrictions.one_way_routes import OneWayRoutes

def run_tsp_with_one_way_routes(
    cities_locations: List[Tuple[float, float]],
    one_way_routes: OneWayRoutes,
    population_size: int = 100,
    n_generations: int = 100,
    mutation_probability: float = 0.3,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Executa o algoritmo genético do TSP com restrições de rotas unidirecionais.
    
    Parâmetros:
    - cities_locations: Lista de coordenadas das cidades
    - one_way_routes: Objeto OneWayRoutes com as rotas unidirecionais
    - population_size: Tamanho da população
    - n_generations: Número de gerações
    - mutation_probability: Probabilidade de mutação
    
    Retorna:
    - Tuple[melhor_solução, melhor_fitness]
    """
    # Cria população inicial
    population = generate_random_population(cities_locations, population_size)

    # Lista para armazenar os melhores valores de fitness
    best_fitness_values = []
    best_solutions = []

    for generation in range(n_generations):
        # Calcular fitness considerando as rotas unidirecionais
        population_fitness = [
            calculate_fitness(individual, one_way_routes)
            for individual in population
        ]
        
        # Ordenar população
        population, population_fitness = sort_population(population, population_fitness)
        
        best_fitness = calculate_fitness(population[0], one_way_routes)
        best_solution = population[0]
        
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)
        
        print(f"Geração {generation}: Melhor fitness = {best_fitness}")
        
        # Elitismo: mantém o melhor indivíduo
        new_population = [population[0]]
        
        # Cria nova população
        while len(new_population) < population_size:
            # Seleção
            # Selecionar dois pais aleatórios dos 10 melhores indivíduos
            indices = random.sample(range(min(10, len(population))), 2)
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]
            
            # Crossover
            child1, child2 = order_crossover(parent1, parent2)
            
            # Mutação
            child1 = mutate(child1, mutation_probability)
            
            new_population.append(child1)
        
        population = new_population
    
    # Retorna a melhor solução encontrada
    return best_solutions[-1], best_fitness_values[-1]

def main():
    """Exemplo de uso da restrição de rotas unidirecionais."""
    # Exemplo de uso
    N_CITIES = 10
    cities_locations = [
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(N_CITIES)
    ]
    
    # Cria algumas rotas unidirecionais para teste
    one_way_routes = OneWayRoutes()
    
    # Adiciona rotas unidirecionais aleatórias (10% das possíveis rotas)
    n_possible_routes = (N_CITIES * (N_CITIES - 1)) // 2
    n_one_way = n_possible_routes // 10  # 10% das rotas
    
    for _ in range(n_one_way):
        # Seleciona duas cidades aleatórias
        indices = random.sample(range(N_CITIES), 2)
        city1 = cities_locations[indices[0]]
        city2 = cities_locations[indices[1]]
        
        # Define uma direção permitida (city1 -> city2)
        one_way_routes.add_one_way_route(city1, city2)
    
    print(f"\nNúmero de rotas unidirecionais: {len(one_way_routes.get_all_one_way_routes())}")
    print("Rotas unidirecionais (permitidas apenas nesta direção):")
    for route in one_way_routes.get_all_one_way_routes():
        print(f"  {route[0]} -> {route[1]}")
    
    # Executa o algoritmo
    best_solution, best_fitness = run_tsp_with_one_way_routes(
        cities_locations=cities_locations,
        one_way_routes=one_way_routes,
        population_size=100,
        n_generations=100,
        mutation_probability=0.3,
    )
    
    print("\nMelhor solução encontrada:")
    print(f"Fitness: {best_fitness:.2f}")
    print("Rota:")
    for city in best_solution:
        print(f"  {city}")
    
    # Verifica se a melhor solução usa alguma rota na contramão
    n = len(best_solution)
    wrong_way_routes_used = []
    for i in range(n):
        city1 = best_solution[i]
        city2 = best_solution[(i + 1) % n]
        if one_way_routes.is_wrong_way(city1, city2):
            wrong_way_routes_used.append((city1, city2))
    
    if wrong_way_routes_used:
        print("\nAtenção: A solução ainda usa as seguintes rotas na contramão:")
        for route in wrong_way_routes_used:
            print(f"  {route[0]} -> {route[1]}")
        print("Considere aumentar a penalidade ou o número de gerações.")
    else:
        print("\nSucesso: A solução não usa nenhuma rota na contramão!")

if __name__ == '__main__':
    main()
