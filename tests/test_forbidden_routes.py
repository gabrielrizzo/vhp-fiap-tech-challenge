#!/usr/bin/env python3
"""
Testes para a restrição de rotas proibidas (forbidden_routes)
"""
import sys
import os

# Adiciona o diretório raiz ao path para importar os módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from restrictions.forbidden_routes import ForbiddenRoutes
from core.restriction_manager import RestrictionManager
from core.enhanced_genetic_algorithm import EnhancedGeneticAlgorithm


def test_forbidden_routes():
    """Testa a restrição de rotas proibidas"""
    print("=== TESTE DA RESTRIÇÃO DE ROTAS PROIBIDAS ===")
    
    # Cria algumas cidades de teste
    cities = [
        (0, 0),     # Cidade A
        (10, 10),   # Cidade B
        (20, 0),    # Cidade C
        (10, -10),  # Cidade D
    ]
    
    # Cria uma rota circular
    test_route = cities + [cities[0]]
    
    # Cria a restrição de rotas proibidas
    forbidden_restriction = ForbiddenRoutes(base_distance_penalty=1000.0)
    
    # Teste 1: Sem rotas proibidas definidas
    print("\n--- Teste 1: Sem rotas proibidas definidas ---")
    is_valid = forbidden_restriction.validate_route(test_route)
    penalty = forbidden_restriction.calculate_penalty(test_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Teste 2: Adiciona uma rota proibida (A -> B)
    print("\n--- Teste 2: Adiciona rota proibida A -> B ---")
    forbidden_restriction.add_forbidden_route(cities[0], cities[1])
    
    # Verifica se a rota A -> B está proibida
    is_forbidden = forbidden_restriction.is_route_forbidden(cities[0], cities[1])
    print(f"A -> B está proibida: {is_forbidden}")
    
    # Verifica se a rota B -> A também está proibida (deve estar, pois é bidirecional)
    is_forbidden = forbidden_restriction.is_route_forbidden(cities[1], cities[0])
    print(f"B -> A está proibida: {is_forbidden}")
    
    # Teste 3: Rota com uma via proibida
    print("\n--- Teste 3: Rota com uma via proibida ---")
    is_valid = forbidden_restriction.validate_route(test_route)
    penalty = forbidden_restriction.calculate_penalty(test_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Teste 4: Rota alternativa sem a via proibida
    print("\n--- Teste 4: Rota alternativa sem a via proibida ---")
    alternative_route = [cities[0], cities[3], cities[2], cities[1], cities[0]]
    is_valid = forbidden_restriction.validate_route(alternative_route)
    penalty = forbidden_restriction.calculate_penalty(alternative_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Teste 5: Obter informações sobre rotas proibidas
    print("\n--- Teste 5: Informações sobre rotas proibidas ---")
    info = forbidden_restriction.get_forbidden_routes_info(test_route)
    print(f"Informações: {info}")
    
    # Teste 6: Integração com RestrictionManager
    print("\n--- Teste 6: Integração com RestrictionManager ---")
    restriction_manager = RestrictionManager()
    restriction_manager.add_restriction(forbidden_restriction)
    
    # Valida rota através do manager
    is_valid_manager = restriction_manager.validate_route(test_route)
    print(f"Validação via manager: {is_valid_manager}")
    
    # Calcula fitness com restrições
    ga = EnhancedGeneticAlgorithm(cities)
    ga.restriction_manager = restriction_manager
    
    fitness = ga.calculate_fitness_with_restrictions(test_route)
    print(f"Fitness com restrições: {fitness}")
    
    # Teste 7: Múltiplas rotas proibidas
    print("\n--- Teste 7: Múltiplas rotas proibidas ---")
    forbidden_restriction.add_forbidden_route(cities[2], cities[3])
    
    # Rota que viola múltiplas restrições
    very_bad_route = [cities[0], cities[1], cities[2], cities[3], cities[0]]
    is_valid = forbidden_restriction.validate_route(very_bad_route)
    penalty = forbidden_restriction.calculate_penalty(very_bad_route)
    print(f"Rota válida: {is_valid}")
    print(f"Penalidade: {penalty}")
    
    # Obter todas as rotas proibidas
    all_forbidden_routes = forbidden_restriction.get_all_forbidden_routes()
    print(f"Todas as rotas proibidas: {all_forbidden_routes}")
    
    # Teste 8: Limpar rotas proibidas
    print("\n--- Teste 8: Limpar rotas proibidas ---")
    forbidden_restriction.clear_forbidden_routes()
    all_forbidden_routes = forbidden_restriction.get_all_forbidden_routes()
    print(f"Rotas proibidas após limpar: {all_forbidden_routes}")
    
    is_valid = forbidden_restriction.validate_route(test_route)
    print(f"Rota válida após limpar: {is_valid}")
    
    print("\n=== TESTE DE ROTAS PROIBIDAS CONCLUÍDO ===")


def test_forbidden_routes_with_config():
    """Testa a integração da restrição de rotas proibidas com o sistema de configuração"""
    print("\n=== TESTE DE INTEGRAÇÃO COM CONFIGURAÇÃO ===")
    
    try:
        from core.config_manager import ConfigManager
        
        # Carrega configuração
        config = ConfigManager()
        
        # Verifica se a configuração de rotas proibidas está presente
        forbidden_routes_config = config.get("restrictions.forbidden_routes", {})
        print(f"Configuração de rotas proibidas: {forbidden_routes_config}")
        
        # Verifica se está habilitada
        is_enabled = forbidden_routes_config.get("enabled", False)
        print(f"Restrição habilitada: {is_enabled}")
        
        if is_enabled:
            weight = forbidden_routes_config.get("weight", 1.0)
            base_penalty = forbidden_routes_config.get("base_distance_penalty", 1000.0)
            print(f"Peso da restrição: {weight}")
            print(f"Penalidade base: {base_penalty}")
        
        print("=== TESTE DE CONFIGURAÇÃO CONCLUÍDO ===")
    except ImportError:
        print("ERROR - ConfigManager não encontrado, pulando teste de configuração")


if __name__ == "__main__":
    test_forbidden_routes()
    test_forbidden_routes_with_config()