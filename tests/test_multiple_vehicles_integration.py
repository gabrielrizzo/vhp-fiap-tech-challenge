#!/usr/bin/env python3
"""
Teste de integração para a restrição de múltiplos veículos
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restrictions.multiple_vehicles import MultipleVehiclesRestriction
from core.restriction_manager import RestrictionManager
from core.enhanced_genetic_algorithm import EnhancedGeneticAlgorithm

def test_multiple_vehicles_restriction():
    """Testa a restrição de múltiplos veículos"""
    print("=== TESTE DA RESTRIÇÃO DE MÚLTIPLOS VEÍCULOS ===")
    
    # Cria algumas cidades de teste
    cities = [
        (0, 0),    # Depósito
        (10, 10),  # Paciente 1
        (20, 20),  # Paciente 2
        (30, 30),  # Paciente 3
        (40, 40),  # Paciente 4
        (50, 50),  # Paciente 5
        (60, 60),  # Paciente 6
        (70, 70),  # Paciente 7
        (80, 80),  # Paciente 8
        (90, 90),  # Paciente 9
        (100, 100) # Paciente 10
    ]
    
    depot = cities[0]
    patients = cities[1:]
    
    # Teste 1: Restrição com 2 veículos e capacidade 1 (deve ser suficiente para 10 pacientes)
    print("\n--- Teste 1: 2 veículos com capacidade 1 para 10 pacientes ---")
    restriction = MultipleVehiclesRestriction(max_vehicles=2, depot=depot, vehicle_capacity=1)
    
    # Cria uma rota que inclui o depósito
    route = [depot] + patients + [depot]
    
    print(f"Rota: {len(route)} cidades")
    print(f"Depósito: {depot}")
    print(f"Pacientes: {len(patients)}")
    
    # Valida a rota
    is_valid = restriction.validate_route(route)
    print(f"Rota válida: {is_valid}")
    
    # Calcula penalidade
    penalty = restriction.calculate_penalty(route)
    print(f"Penalidade: {penalty}")
    
    # Obtém informações detalhadas
    info = restriction.get_multiple_vehicles_info(route)
    print(f"Informações: {info}")
    
    # Teste 2: Restrição com 1 veículo e capacidade 1 (deve ser insuficiente para 10 pacientes)
    print("\n--- Teste 2: 1 veículo com capacidade 1 para 10 pacientes ---")
    restriction2 = MultipleVehiclesRestriction(max_vehicles=1, depot=depot, vehicle_capacity=1)
    
    is_valid2 = restriction2.validate_route(route)
    print(f"Rota válida: {is_valid2}")
    
    penalty2 = restriction2.calculate_penalty(route)
    print(f"Penalidade: {penalty2}")
    
    # Teste 3: Integração com RestrictionManager
    print("\n--- Teste 3: Integração com RestrictionManager ---")
    restriction_manager = RestrictionManager()
    restriction_manager.add_restriction(restriction)
    
    # Valida rota através do manager
    is_valid_manager = restriction_manager.validate_route(route)
    print(f"Validação via manager: {is_valid_manager}")
    
    # Calcula fitness com restrições
    ga = EnhancedGeneticAlgorithm(cities)
    ga.restriction_manager = restriction_manager
    
    fitness = ga.calculate_fitness_with_restrictions(route)
    print(f"Fitness com restrições: {fitness}")
    
    # Teste 4: Estatísticas da população
    print("\n--- Teste 4: Estatísticas da população ---")
    population = [route, route, route]  # População de teste
    stats = ga.get_population_statistics(population)
    print(f"Estatísticas: {stats}")
    
    # Teste 5: Teste com capacidade diferente (5 pacientes por veículo)
    print("\n--- Teste 5: 2 veículos com capacidade 5 para 10 pacientes ---")
    restriction5 = MultipleVehiclesRestriction(max_vehicles=2, depot=depot, vehicle_capacity=5)
    
    is_valid5 = restriction5.validate_route(route)
    print(f"Rota válida: {is_valid5}")
    
    penalty5 = restriction5.calculate_penalty(route)
    print(f"Penalidade: {penalty5}")
    
    info5 = restriction5.get_multiple_vehicles_info(route)
    print(f"Informações: {info5}")
    
    print("\n=== TESTE CONCLUÍDO ===")

def test_configuration_integration():
    """Testa a integração com o sistema de configuração"""
    print("\n=== TESTE DE INTEGRAÇÃO COM CONFIGURAÇÃO ===")
    
    from core.config_manager import ConfigManager
    
    # Carrega configuração
    config = ConfigManager()
    
    # Verifica se a configuração de múltiplos veículos está presente
    multiple_vehicles_config = config.get("restrictions.multiple_vehicles", {})
    print(f"Configuração de múltiplos veículos: {multiple_vehicles_config}")
    
    # Verifica se está habilitada
    is_enabled = multiple_vehicles_config.get("enabled", False)
    print(f"Restrição habilitada: {is_enabled}")
    
    if is_enabled:
        max_vehicles = multiple_vehicles_config.get("max_vehicles", 5)
        weight = multiple_vehicles_config.get("weight", 2.0)
        print(f"Máximo de veículos: {max_vehicles}")
        print(f"Peso da restrição: {weight}")
    
    print("=== TESTE DE CONFIGURAÇÃO CONCLUÍDO ===")

if __name__ == "__main__":
    test_multiple_vehicles_restriction()
    test_configuration_integration()
