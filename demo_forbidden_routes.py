import random
from typing import List, Tuple
from genetic_algorithm import (
    generate_random_population,
    calculate_fitness,
    order_crossover,
    mutate,
    sort_population
)
from restrictions.forbidden_routes import ForbiddenRoutes

def run_tsp_with_forbidden_routes(
    cities_locations: List[Tuple[float, float]],
    forbidden_routes: ForbiddenRoutes,
    population_size: int = 100,
    n_generations: int = 100,
    mutation_probability: float = 0.3,
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Executa o algoritmo genético do TSP com restrições de rotas proibidas.

    Parâmetros:
    - cities_locations: Lista de coordenadas das cidades
    - forbidden_routes: Objeto ForbiddenRoutes com as rotas proibidas
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
        # Calcula o fitness considerando as rotas proibidas
        population_fitness = [
            calculate_fitness(individual, forbidden_routes)
            for individual in population
        ]

        # Ordena a população pelo fitness
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(population[0], forbidden_routes)
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Geração {generation}: Melhor fitness = {best_fitness}")

        # Elitismo: mantém o melhor indivíduo
        new_population = [population[0]]

        # Cria nova população
        while len(new_population) < population_size:
            # Seleção
            parent1, parent2 = random.choices(population[:10], k=2)

            # Crossover
            child1, child2 = order_crossover(parent1, parent2)

            # Mutação
            child1 = mutate(child1, mutation_probability)

            new_population.append(child1)

        population = new_population
    
    # Retorna a melhor solução encontrada
    return best_solutions[-1], best_fitness_values[-1]

def main() -> None:
    """Exemplo de uso da restrição de rotas proibidas."""
    # Exemplo de uso
    N_CITIES = 10
    cities_locations = [
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(N_CITIES)
    ]

    # Cria algumas rotas proibidas para teste
    forbidden_routes = ForbiddenRoutes()

    # Proíbe algumas rotas aleatórias (20% das possíveis rotas)
    n_possible_routes = (N_CITIES * (N_CITIES - 1)) // 2
    n_forbidden = n_possible_routes // 5  # 20% das rotas

    for _ in range(n_forbidden):
        city1, city2 = random.sample(cities_locations, 2)
        forbidden_routes.add_forbidden_route(city1, city2)

    print(f"\nNúmero de rotas proibidas: {len(forbidden_routes.get_all_forbidden_routes())}")
    print("Rotas proibidas:")
    for route in forbidden_routes.get_all_forbidden_routes():
        print(f"  {route[0]} -> {route[1]}")

    # Executa o algoritmo
    best_solution, best_fitness = run_tsp_with_forbidden_routes(
        cities_locations=cities_locations,
        forbidden_routes=forbidden_routes,
        population_size=100,
        n_generations=100,
        mutation_probability=0.3,
    )

    print("\nMelhor solução encontrada:")
    print(f"Fitness: {best_fitness}")
    print("Rota:")
    for city in best_solution:
        print(f"  {city}")

    # Verifica se a melhor solução usa alguma rota proibida
    n = len(best_solution)
    forbidden_routes_used = []
    for i in range(n):
        city1 = best_solution[i]
        city2 = best_solution[(i + 1) % n]
        if forbidden_routes.is_route_forbidden(city1, city2):
            forbidden_routes_used.append((city1, city2))

    if forbidden_routes_used:
        print("\nAtenção: A solução ainda usa as seguintes rotas proibidas:")
        for route in forbidden_routes_used:
            print(f"  {route[0]} -> {route[1]}")
        print("Considere aumentar a penalidade ou o número de gerações.")
    else:
        print("\nSucesso: A solução não usa nenhuma rota proibida!")

if __name__ == '__main__':
    main()
