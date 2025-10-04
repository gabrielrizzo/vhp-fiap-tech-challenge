#!/usr/bin/env python3
"""
Testes para a restrição de rotas unidirecionais (one_way_routes)
"""
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restrictions.one_way_routes import OneWayRoutes
from core.restriction_manager import RestrictionManager
from core.enhanced_genetic_algorithm import EnhancedGeneticAlgorithm


def test_one_way_routes():
    """Testa a restrição de rotas unidirecionais (mão única)"""
    print("=== TESTE DA RESTRIÇÃO DE ROTAS UNIDIRECIONAIS ===")
    
    # Cria algumas cidades de teste
    cities = [
        (0, 0),     # Cidade A
        (10, 10),   # Cidade B
        (20, 0),    # Cidade C
        (10, -10),  # Cidade D
    ]
    
    # Cria uma rota circular
    route = cities + [cities[0]]
    
    # Cria a restrição de rotas unidirecionais
    one_way_restriction = OneWayRoutes(base_distance_penalty=1000.0)
    
    # Teste 1: Sem rotas unidirecionais definidas
    print("\n--- Teste 1: Sem rotas unidirecionais definidas ---")
    is_valid = one_way_restriction.validate_route(route)
    penalty = one_way_restriction.calculate_penalty(route)
    print(f"Rota válida: {is_valid}") 
    print(f"Penalidade: {penalty}") 
    
    # Teste 2: Adiciona uma rota unidirecional (A -> B)
    print("\n--- Teste 2: Adiciona rota unidirecional A -> B ---")
    one_way_restriction.add_one_way_route(cities[0], cities[1])
    
    # Verifica se a rota B -> A está na contramão
    is_wrong_way = one_way_restriction.is_wrong_way(cities[1], cities[0])
    print(f"B -> A está na contramão: {is_wrong_way}")
    
    # Teste 3: Rota com uma via de mão única na direção correta
    print("\n--- Teste 3: Rota com uma via de mão única na direção correta ---")
    correct_route = [cities[0], cities[1], cities[2], cities[3], cities[0]]
    is_valid = one_way_restriction.validate_route(correct_route)
    penalty = one_way_restriction.calculate_penalty(correct_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Teste 4: Rota com uma via de mão única na direção errada
    print("\n--- Teste 4: Rota com uma via de mão única na direção errada ---")
    wrong_route = [cities[3], cities[2], cities[1], cities[0], cities[3]]
    is_valid = one_way_restriction.validate_route(wrong_route)
    penalty = one_way_restriction.calculate_penalty(wrong_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Teste 5: Obter informações sobre rotas em contramão
    print("\n--- Teste 5: Informações sobre rotas em contramão ---")
    info = one_way_restriction.get_wrong_way_routes(wrong_route)
    print(f"Informações: {info}")
    
    # Teste 6: Integração com RestrictionManager
    print("\n--- Teste 6: Integração com RestrictionManager ---")
    restriction_manager = RestrictionManager()
    restriction_manager.add_restriction(one_way_restriction)
    
    # Valida rota através do manager
    is_valid_manager = restriction_manager.validate_route(wrong_route)
    print(f"Validação via manager: {is_valid_manager}")
    
    # Calcula fitness com restrições
    ga = EnhancedGeneticAlgorithm(cities)
    ga.restriction_manager = restriction_manager
    
    fitness = ga.calculate_fitness_with_restrictions(wrong_route)
    print(f"Fitness com restrições: {fitness}")
    
    # Teste 7: Múltiplas rotas unidirecionais
    print("\n--- Teste 7: Múltiplas rotas unidirecionais ---")
    one_way_restriction.add_one_way_route(cities[2], cities[3])
    
    # Rota que viola múltiplas restrições de mão única
    very_wrong_route = [cities[1], cities[0], cities[3], cities[2], cities[1]]
    is_valid = one_way_restriction.validate_route(very_wrong_route)
    penalty = one_way_restriction.calculate_penalty(very_wrong_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Obter todas as rotas unidirecionais
    all_one_way_routes = one_way_restriction.get_all_one_way_routes()
    print(f"Todas as rotas unidirecionais: {all_one_way_routes}")
    
    # Teste 8: Limpar rotas unidirecionais
    print("\n--- Teste 8: Limpar rotas unidirecionais ---")
    one_way_restriction.clear_one_way_routes()
    all_one_way_routes = one_way_restriction.get_all_one_way_routes()
    print(f"Rotas unidirecionais após limpar: {all_one_way_routes}")
    
    is_valid = one_way_restriction.validate_route(wrong_route)
    print(f"Rota válida após limpar: {is_valid}")
    
    print("\n=== TESTE DE ROTAS UNIDIRECIONAIS CONCLUÍDO ===")


def test_one_way_routes_with_config():
    """Testa a integração da restrição de rotas unidirecionais com o sistema de configuração"""
    print("\n=== TESTE DE INTEGRAÇÃO COM CONFIGURAÇÃO ===")
    
    try:
        from core.config_manager import ConfigManager
        
        # Carrega configuração
        config = ConfigManager()
        
        # Verifica se a configuração de rotas unidirecionais está presente
        one_way_routes_config = config.get("restrictions.one_way_routes", {})
        print(f"Configuração de rotas unidirecionais: {one_way_routes_config}")
        
        # Verifica se está habilitada
        is_enabled = one_way_routes_config.get("enabled", False)
        print(f"Restrição habilitada: {is_enabled}")
        
        if is_enabled:
            weight = one_way_routes_config.get("weight", 1.0)
            base_penalty = one_way_routes_config.get("base_distance_penalty", 2000.0)
            print(f"Peso da restrição: {weight}")
            print(f"Penalidade base: {base_penalty}")
        
        print("=== TESTE DE CONFIGURAÇÃO CONCLUÍDO ===")
    except ImportError:
        print("ERROR - ConfigManager não encontrado, pulando teste de configuração")


def test_integration_with_other_restrictions():
    """Testa a integração da restrição de rotas unidirecionais com outras restrições"""
    print("\n=== TESTE DE INTEGRAÇÃO COM OUTRAS RESTRIÇÕES ===")
    
    try:
        from restrictions.forbidden_routes import ForbiddenRoutes
        
        # Cria algumas cidades de teste
        cities = [
            (0, 0),     # Cidade A
            (10, 10),   # Cidade B
            (20, 0),    # Cidade C
            (10, -10),  # Cidade D
        ]
        
        # Cria as restrições
        one_way_restriction = OneWayRoutes(base_distance_penalty=1000.0)
        forbidden_restriction = ForbiddenRoutes(base_distance_penalty=1000.0)
        
        # Adiciona algumas restrições
        one_way_restriction.add_one_way_route(cities[0], cities[1])  # A -> B é mão única
        forbidden_restriction.add_forbidden_route(cities[2], cities[3])  # C -> D é proibida
        
        # Adiciona as restrições ao manager
        restriction_manager = RestrictionManager()
        restriction_manager.add_restriction(one_way_restriction)
        restriction_manager.add_restriction(forbidden_restriction)
        
        # Teste 1: Rota que viola ambas as restrições
        print("\n--- Teste 1: Rota que viola ambas as restrições ---")
        bad_route = [cities[1], cities[0], cities[2], cities[3], cities[1]]
        
        is_valid = restriction_manager.validate_route(bad_route)
        print(f"Rota válida: {is_valid}")
        
        # Obtém resumo das violações
        violation_summary = restriction_manager.get_violation_summary(bad_route)
        print(f"Resumo das violações: {violation_summary}")
        
        # Teste 2: Rota que não viola nenhuma restrição
        print("\n--- Teste 2: Rota que não viola nenhuma restrição ---")
        good_route = [cities[0], cities[1], cities[2], cities[0]]
        
        is_valid = restriction_manager.validate_route(good_route)
        print(f"OK - Rota válida: {is_valid}")
        
        violation_summary = restriction_manager.get_violation_summary(good_route)
        print(f"Resumo das violações: {violation_summary}")
        
        # Teste 3: Calcular fitness com ambas as restrições
        print("\n--- Teste 3: Calcular fitness com ambas as restrições ---")
        ga = EnhancedGeneticAlgorithm(cities)
        ga.restriction_manager = restriction_manager
        
        bad_fitness = ga.calculate_fitness_with_restrictions(bad_route)
        good_fitness = ga.calculate_fitness_with_restrictions(good_route)
        
        print(f"Fitness da rota ruim: {bad_fitness}")
        print(f"Fitness da rota boa: {good_fitness}")
        
        print("=== TESTE DE INTEGRAÇÃO COM OUTRAS RESTRIÇÕES CONCLUÍDO ===")
    except ImportError:
        print("ERROR - ForbiddenRoutes não encontrado, pulando teste de integração")


if __name__ == "__main__":
    test_one_way_routes()
    test_one_way_routes_with_config()
    test_integration_with_other_restrictions()