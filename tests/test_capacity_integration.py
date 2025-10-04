#!/usr/bin/env python3
"""
Teste de integração entre VehicleCapacityRestriction e MultipleVehiclesRestriction
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction
from restrictions.multiple_vehicles import MultipleVehiclesRestriction
from core.restriction_manager import RestrictionManager

def test_capacity_integration():
    """Testa a integração entre as duas restrições"""
    print("=== TESTE DE INTEGRAÇÃO DE CAPACIDADE ===")
    
    # Cria algumas cidades de teste
    cities = [
        (0, 0),    # Depósito
        (10, 10),  # Paciente 1
        (20, 20),  # Paciente 2
        (30, 30),  # Paciente 3
        (40, 40),  # Paciente 4
        (50, 50),  # Paciente 5
    ]
    
    depot = cities[0]
    patients = cities[1:]
    
    # Cria as restrições
    capacity_restriction = VehicleCapacityRestriction(max_patients_per_vehicle=3)
    multiple_vehicles_restriction = MultipleVehiclesRestriction(
        max_vehicles=2, 
        depot=depot, 
        vehicle_capacity=3
    )
    
    # Cria uma rota que inclui o depósito
    route = [depot] + patients + [depot]
    
    print(f"Rota: {len(route)} cidades")
    print(f"Depósito: {depot}")
    print(f"Pacientes: {len(patients)}")
    print(f"Capacidade por veículo: 3")
    print(f"Máximo de veículos: 2")
    
    # Teste 1: Validação individual das restrições
    print("\n--- Teste 1: Validação individual ---")
    
    # Dados do veículo para integração
    vehicle_data = multiple_vehicles_restriction.get_vehicle_data_for_capacity_restriction()
    print(f"Dados do veículo: {vehicle_data}")
    
    # Validação da restrição de capacidade
    capacity_valid = capacity_restriction.validate_route(route, vehicle_data)
    capacity_penalty = capacity_restriction.calculate_penalty(route, vehicle_data)
    print(f"Restrição de capacidade - Válida: {capacity_valid}, Penalidade: {capacity_penalty}")
    
    # Validação da restrição de múltiplos veículos
    multiple_valid = multiple_vehicles_restriction.validate_route(route)
    multiple_penalty = multiple_vehicles_restriction.calculate_penalty(route)
    print(f"Restrição de múltiplos veículos - Válida: {multiple_valid}, Penalidade: {multiple_penalty}")
    
    # Teste 2: Integração com RestrictionManager
    print("\n--- Teste 2: Integração com RestrictionManager ---")
    
    restriction_manager = RestrictionManager()
    restriction_manager.add_restriction(capacity_restriction)
    restriction_manager.add_restriction(multiple_vehicles_restriction)
    
    # Validação via manager
    manager_valid = restriction_manager.validate_route(route, vehicle_data)
    print(f"Validação via manager: {manager_valid}")
    
    # Resumo de violações
    violation_summary = restriction_manager.get_violation_summary(route, vehicle_data)
    print(f"Resumo de violações: {violation_summary}")
    
    # Teste 3: Informações detalhadas
    print("\n--- Teste 3: Informações detalhadas ---")
    
    capacity_info = capacity_restriction.get_capacity_info(route, vehicle_data)
    print(f"Informações de capacidade: {capacity_info}")
    
    multiple_info = multiple_vehicles_restriction.get_multiple_vehicles_info(route)
    print(f"Informações de múltiplos veículos: {multiple_info}")
    
    # Teste 4: Cenário com excesso de pacientes
    print("\n--- Teste 4: Cenário com excesso de pacientes ---")
    
    # Cria uma rota com mais pacientes que a capacidade permite
    excess_patients = [(i*10, i*10) for i in range(1, 8)]  # 7 pacientes
    excess_route = [depot] + excess_patients + [depot]
    
    print(f"Rota com excesso: {len(excess_route)} cidades, {len(excess_patients)} pacientes")
    
    excess_capacity_valid = capacity_restriction.validate_route(excess_route, vehicle_data)
    excess_capacity_penalty = capacity_restriction.calculate_penalty(excess_route, vehicle_data)
    print(f"Restrição de capacidade - Válida: {excess_capacity_valid}, Penalidade: {excess_capacity_penalty}")
    
    excess_multiple_valid = multiple_vehicles_restriction.validate_route(excess_route)
    excess_multiple_penalty = multiple_vehicles_restriction.calculate_penalty(excess_route)
    print(f"Restrição de múltiplos veículos - Válida: {excess_multiple_valid}, Penalidade: {excess_multiple_penalty}")
    
    print("\n=== TESTE DE INTEGRAÇÃO CONCLUÍDO ===")

if __name__ == "__main__":
    test_capacity_integration()
