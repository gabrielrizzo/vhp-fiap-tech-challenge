import random
from benchmark_att48 import att_48_cities_locations
from typing import List, Tuple
from genetic_algorithm import (
    generate_random_population,
    calculate_fitness,
    order_crossover,
    mutate,
    sort_population
)
from restrictions.multiple_vehicles.multiple_vehicles import MultipleVehicles

def run_tsp_with_multiple_vehicles_demo(
    patients: List[Tuple[float, float]],
    depot: Tuple[float, float],
    max_vehicles: int = 5,
    vehicle_capacity: int = 10,
    population_size: int = 100,
    n_generations: int = 100,
    mutation_probability: float = 0.3,
) -> Tuple[List[Tuple[float, float]], float, dict]:
    """
    Executa o algoritmo genético do TSP com restrições de múltiplos veículos.

    Parâmetros:
    - patients: Lista de coordenadas (x, y) dos pacientes a serem atendidos
    - depot: Coordenadas (x, y) do depósito ou hospital base
    - multiple_vehicles: Objeto MultipleVehicles configurado com capacidades
    - vehicle_capacity: Capacidade de cada ambulância
    - population_size: Tamanho da população
    - n_generations: Número de gerações
    - mutation_probability: Probabilidade de mutação

    Retorna:
    - Tuple[melhor_solução, melhor_fitness, métricas_detalhadas]
    """
    # Cria população inicial
    population = generate_random_population(patients, population_size)

    # Lista para armazenar os melhores valores de fitness
    best_fitness_values = []
    best_solutions = []

    for generation in range(n_generations):
        # Calcula o fitness considerando múltiplos veículos
        population_fitness = []
        for individual in population:
            # Cria objeto de múltiplos veículos para cada indivíduo
            multiple_vehicles = MultipleVehicles(max_vehicles, vehicle_capacity)
            fitness = calculate_fitness(individual, multiple_vehicles, depot)
            population_fitness.append(fitness)

        # Ordena a população pelo fitness
        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Geração {generation}: Melhor fitness = {best_fitness:.2f}")

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
    
    # Calcula métricas finais da melhor solução
    best_multiple_vehicles = MultipleVehicles(max_vehicles, vehicle_capacity)
    best_multiple_vehicles.assign_patients_to_vehicles(best_solution, depot)
    
    metrics = {
        'total_time': best_multiple_vehicles.get_total_time(),
        'average_time': best_multiple_vehicles.get_average_time(),
        'vehicle_count': best_multiple_vehicles.get_vehicle_count(),
        'vehicle_utilization': best_multiple_vehicles.get_vehicle_utilization(),
        'vehicle_routes': best_multiple_vehicles.vehicle_routes
    }
    
    # Retorna a melhor solução encontrada
    return best_solutions[-1], best_fitness_values[-1], metrics

def generate_random_patients(n_patients: int, x_range: tuple = (0, 100), y_range: tuple = (0, 100)) -> List[Tuple[float, float]]:
    """Gera pacientes aleatórios dentro de um range específico."""
    return [
        (random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))
        for _ in range(n_patients)
    ]

def load_att48_patients() -> List[Tuple[float, float]]:
    """Carrega pacientes do dataset ATT48 (48 cidades)."""
    return att_48_cities_locations

"""
Demo: Múltiplas Ambulâncias para TSP Médico

Este demo demonstra o uso do sistema de múltiplas ambulâncias para otimização
de rotas de atendimento médico usando algoritmo genético.

CONFIGURAÇÕES DISPONÍVEIS:
1. PATIENT_SOURCE: Escolha entre:
   - "random": Gera pacientes aleatórios
   - "att48": Usa dataset ATT48 (48 cidades)

2. Para pacientes aleatórios (PATIENT_SOURCE == "random"):
   - N_PATIENTS: Número de pacientes
   - X_RANGE, Y_RANGE: Ranges de coordenadas

3. Para ambulâncias:
   - MAX_VEHICLES: Número máximo de ambulâncias
   - VEHICLE_CAPACITY: Capacidade de cada ambulância

4. Para algoritmo genético:
   - POPULATION_SIZE: Tamanho da população
   - N_GENERATIONS: Número de gerações
   - MUTATION_PROBABILITY: Probabilidade de mutação
"""
def main() -> None:
    """Exemplo de uso da restrição de múltiplos veículos."""
    
    # ===== CONFIGURAÇÕES =====
    # Configurações do sistema de ambulâncias
    MAX_VEHICLES = 20 # Número máximo de ambulâncias
    VEHICLE_CAPACITY = 8  # Capacidade de cada ambulância

    # Escolha o tipo de geração de pacientes:
    PATIENT_SOURCE = "att48"  # Opções: "random", "att48"
    
    # ===== GERAÇÃO DE PACIENTES =====
    if PATIENT_SOURCE == "random":
        # Configurações para pacientes aleatórios (PATIENT_SOURCE == "random")
        N_PATIENTS = 20
        X_RANGE = (0, 100)  # Range X para pacientes aleatórios
        Y_RANGE = (0, 100)  # Range Y para pacientes aleatórios

        patients = generate_random_patients(N_PATIENTS, X_RANGE, Y_RANGE)
        print(f"Usando {N_PATIENTS} pacientes aleatórios")
        depot = (50, 50)  # Centro padrão
    elif PATIENT_SOURCE == "att48":
        patients = load_att48_patients()
        print(f"Usando {len(patients)} pacientes do dataset ATT48")
        depot = (4000, 2000)  # Centro das coordenadas ATT48
    else:
        raise ValueError("PATIENT_SOURCE deve ser 'random' ou 'att48'")
    
    print(f"Número de pacientes: {len(patients)}")
    print(f"Depósito (hospital): {depot}")
    print(f"Máximo de ambulâncias: {MAX_VEHICLES}")
    print(f"Capacidade por ambulância: {VEHICLE_CAPACITY}")
    print(f"Pacientes: {patients}")
    print()
    
    # ===== CONFIGURAÇÕES DO ALGORITMO GENÉTICO =====
    POPULATION_SIZE = 100
    N_GENERATIONS = 1000
    MUTATION_PROBABILITY = 0.3

    # Executa o algoritmo
    best_solution, best_fitness, metrics = run_tsp_with_multiple_vehicles_demo(
        patients=patients,
        depot=depot,
        max_vehicles=MAX_VEHICLES,
        vehicle_capacity=VEHICLE_CAPACITY,
        population_size=POPULATION_SIZE,
        n_generations=N_GENERATIONS,
        mutation_probability=MUTATION_PROBABILITY,
    )

    print("\n" + "="*60)
    print("RESULTADOS DA OTIMIZAÇÃO")
    print("="*60)
    print(f"Melhor fitness encontrado: {best_fitness:.2f}")
    print(f"Tempo total (ambulância mais lenta): {metrics['total_time']:.2f} minutos")
    print(f"Tempo médio das ambulâncias: {metrics['average_time']:.2f} minutos")
    print(f"Número de ambulâncias utilizadas: {metrics['vehicle_count']}")
    print()
    
    print("UTILIZAÇÃO DAS AMBULÂNCIAS:")
    for vehicle_id, utilization in metrics['vehicle_utilization'].items():
        print(f"  Ambulância {vehicle_id}: {utilization:.1%} utilizada")
    print()
    
    print("ROTAS DETALHADAS:")
    for vehicle_id, route in metrics['vehicle_routes'].items():
        print(f"  Ambulância {vehicle_id}:")
        print(f"    Rota: {route}")
        print(f"    Pacientes atendidos: {len(route) - 2}")  # -2 para excluir depósito inicial e final
        print(f"    Tempo da rota: {metrics['total_time']:.2f} minutos")
        print()
    
    # Análise de eficiência
    print("ANÁLISE DE EFICIÊNCIA:")
    total_patients = len(patients)
    vehicles_used = metrics['vehicle_count']
    avg_utilization = sum(metrics['vehicle_utilization'].values()) / vehicles_used
    
    print(f"  Total de pacientes: {total_patients}")
    print(f"  Ambulâncias utilizadas: {vehicles_used}")
    print(f"  Utilização média: {avg_utilization:.1%}")
    print(f"  Eficiência do sistema: {total_patients / (vehicles_used * VEHICLE_CAPACITY):.1%}")
    
    # Recomendações
    print("\nRECOMENDAÇÕES:")
    if avg_utilization < 0.7:
        print("  - Considerar reduzir o número de ambulâncias para melhorar a utilização")
    elif avg_utilization > 1:
        print("  - Considerar aumentar o número de ambulâncias para reduzir o tempo de atendimento")
    else:
        print("  - Configuração atual está bem balanceada")
    
    if metrics['total_time'] > 120:  # Mais de 2 horas
        print("  - Tempo de atendimento está alto, considere otimizar as rotas ou aumentar o número de ambulâncias")
    else:
        print("  - Tempo de atendimento está dentro de parâmetros aceitáveis")

if __name__ == '__main__':
    main()
