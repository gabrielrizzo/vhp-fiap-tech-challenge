import random
from typing import List, Tuple, Dict
from genetic_algorithm import (
    generate_random_population,
    calculate_fitness,
    order_crossover,
    mutate,
    sort_population
)
from restrictions.forbidden_routes import ForbiddenRoutes
from restrictions.one_way_routes import OneWayRoutes
from restrictions.fixed_start_city import FixedStartCity
from restrictions.restriction_interface import RestrictionInterface


class MultiRestrictionTSP:
    """Classe para gerenciar múltiplas restrições no TSP."""
    def __init__(self):
        """Inicializa o gerenciador de restrições."""
        self.restrictions: Dict[str, RestrictionInterface] = {}
        self.active_restrictions: Dict[str, bool] = {}
    
    def add_restriction(self, name: str, restriction: RestrictionInterface) -> None:
        """Adiciona uma restrição ao gerenciador."""
        self.restrictions[name] = restriction
        self.active_restrictions[name] = False
    
    def activate_restriction(self, name: str) -> None:
        """Ativa uma restrição."""
        if name in self.restrictions:
            self.active_restrictions[name] = True
    
    def deactivate_restriction(self, name: str) -> None:
        """Desativa uma restrição."""
        if name in self.restrictions:
            self.active_restrictions[name] = False
    
    def is_restriction_active(self, name: str) -> bool:
        """Verifica se uma restrição está ativa."""
        return self.active_restrictions.get(name, False)
    
    def get_active_restrictions(self) -> List[str]:
        """Retorna a lista de restrições ativas."""
        return [name for name, active in self.active_restrictions.items() if active]
    
    def calculate_fitness(self, path: List[Tuple[float, float]]) -> float:
        """Calcula o fitness considerando todas as restrições ativas."""
        # Calcula o fitness base
        total_fitness = calculate_fitness(path)
        
        # Adiciona penalidades de cada restrição ativa
        for name, restriction in self.restrictions.items():
            if self.active_restrictions[name]:
                penalty = restriction.fitness_restriction(path)
                total_fitness += penalty
        return total_fitness
    
    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """Verifica se o caminho é válido para todas as restrições ativas."""
        for name, restriction in self.restrictions.items():
            if self.active_restrictions[name] and not restriction.is_valid(path):
                return False
        return True


def run_tsp_with_multiple_restrictions(
    cities_locations: List[Tuple[float, float]],
    multi_restriction: MultiRestrictionTSP,
    population_size: int = 100,
    n_generations: int = 100,
    mutation_probability: float = 0.3,
) -> Tuple[List[Tuple[float, float]], float]:
    """Executa o algoritmo genético do TSP com múltiplas restrições."""
    # Cria população inicial
    population = generate_random_population(cities_locations, population_size)
    
    # Lista para armazenar os melhores valores de fitness
    best_fitness_values = []
    best_solutions = []
    
    for generation in range(n_generations):
        # Calcular fitness considerando todas as restrições ativas
        population_fitness = [
            multi_restriction.calculate_fitness(individual)
            for individual in population
        ]
        
        # Ordenar população
        population, population_fitness = sort_population(population, population_fitness)
        
        best_fitness = multi_restriction.calculate_fitness(population[0])
        best_solution = population[0]
        
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)
        
        print(f"Geração {generation}: Melhor fitness = {best_fitness}")
        
        # Elitismo: mantém o melhor indivíduo
        new_population = [population[0]]
        
        # Cria nova população
        while len(new_population) < population_size:
            # Seleção
            indices = random.sample(range(min(10, len(population))), 2)
            parent1 = population[indices[0]]
            parent2 = population[indices[1]]
            
            # Crossover e mutação
            child = order_crossover(parent1, parent2)[0]
            if random.random() < mutation_probability:
                child = mutate(child, mutation_probability)
            new_population.append(child)
        
        population = new_population
    
    # Retorna a melhor solução encontrada
    return best_solutions[-1], best_fitness_values[-1]


def main():
    """Exemplo de uso com múltiplas restrições."""
    # Configurações
    N_CITIES = 10
    cities_locations = [
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(N_CITIES)
    ]
    
    # Criar gerenciador de restrições
    multi_restriction = MultiRestrictionTSP()
    
    # Criar e configurar restrição de rotas proibidas
    forbidden_routes = ForbiddenRoutes()
    n_forbidden = (N_CITIES * (N_CITIES - 1)) // 10  # 10% das rotas possíveis
    for _ in range(n_forbidden):
        indices = random.sample(range(N_CITIES), 2)
        city1 = cities_locations[indices[0]]
        city2 = cities_locations[indices[1]]
        forbidden_routes.add_forbidden_route(city1, city2)
    
    # Criar e configurar restrição de rotas unidirecionais
    one_way_routes = OneWayRoutes()
    n_one_way = (N_CITIES * (N_CITIES - 1)) // 10  # 10% das rotas possíveis
    for _ in range(n_one_way):
        indices = random.sample(range(N_CITIES), 2)
        city1 = cities_locations[indices[0]]
        city2 = cities_locations[indices[1]]
        one_way_routes.add_one_way_route(city1, city2)
    
    # Criar e configurar restrição de cidade inicial fixa
    fixed_start = FixedStartCity()
    # Define a primeira cidade como ponto inicial obrigatório
    fixed_start.set_start_city(cities_locations[0])
    
    # Adicionar restrições ao gerenciador
    multi_restriction.add_restriction("forbidden_routes", forbidden_routes)
    multi_restriction.add_restriction("one_way_routes", one_way_routes)
    multi_restriction.add_restriction("fixed_start", fixed_start)
    
    # Menu para ativar/desativar restrições
    print("\nRestrições disponíveis:")
    for name, restriction in multi_restriction.restrictions.items():
        print(f"- {name}: {restriction.get_description()}")
    
    print("\nSelecione as restrições que deseja ativar (separadas por vírgula):")
    print("Exemplo: forbidden_routes,one_way_routes,fixed_start")
    selected = input().strip().split(",")
    
    # Ativar restrições selecionadas
    for name in selected:
        name = name.strip()
        if name in multi_restriction.restrictions:
            multi_restriction.activate_restriction(name)
            print(f"Restrição '{name}' ativada!")
    
    print("\nRestrições ativas:", multi_restriction.get_active_restrictions())
    
    # Executar algoritmo
    best_solution, best_fitness = run_tsp_with_multiple_restrictions(
        cities_locations=cities_locations,
        multi_restriction=multi_restriction,
        population_size=100,
        n_generations=100,
        mutation_probability=0.3,
    )
    
    print("\nMelhor solução encontrada:")
    print(f"Fitness: {best_fitness:.2f}")
    print("Rota:")
    for city in best_solution:
        print(f"  {city}")
    
    # Verificar violações de restrições
    print("\nVerificando violações de restrições:")
    for name, restriction in multi_restriction.restrictions.items():
        if multi_restriction.is_restriction_active(name):
            if restriction.is_valid(best_solution):
                print(f"✅ {name}: Solução respeita a restrição")
            else:
                print(f"❌ {name}: Solução viola a restrição")


if __name__ == '__main__':
    main()