"""
    Implementação de classe do custo das rotas
"""
from typing import List, Tuple, Dict
from genetic_algorithm import calculate_fitness

from genetic_algorithm import calculate_distance, calculate_distance_in_km

class RouteCostRestriction:
    """
        Implementação de classe do custo das rotas
        cities_locations: Listagem do problema
        city_route_cost: Dicionário de custos de rotas no formato ((x,y), (x,y)): custo
    """
    def __init__(
            self,
            cities_locations: List[Tuple[float, float]],
            city_route_cost: Dict,
            gas_cost_per_km: float,
            vehicle_quantity: int = 1,
            pixels_to_km_factor: float = 0.01
    ):
        self.cities_locations = cities_locations
        self.city_route_cost = city_route_cost
        self.plot_x_offset = 0
        self.width = 0
        self.height = 0
        self.node_radius = 0
        self.normalized_cities = None
        self.gas_cost_per_km = gas_cost_per_km
        self.vehicle_quantity = vehicle_quantity
        self.pixels_to_km_factor = pixels_to_km_factor

    def config_dimensions(
        self,
        width: float,
        plot_x_offset: float,
        height: float,
        node_radius: float
    ):
        """
            Configura dimensões para possível conversão. Só utilizar se necessiário
            como é o caso do att48
        """
        self.plot_x_offset = plot_x_offset
        self.width = width
        self.height = height
        self.node_radius = node_radius

    def normalize_coordinates(self, original_point: Tuple[float, float]) -> Tuple[int, int]:
        """
        Normaliza coordenadas originais para o sistema de visualização.
        """
        max_x = max(point[0] for point in self.cities_locations)
        max_y = max(point[1] for point in self.cities_locations)
        scale_x = (self.width - self.plot_x_offset - self.node_radius) / max_x
        scale_y = self.height / max_y

        normalized_x = int(original_point[0] * scale_x + self.plot_x_offset)
        normalized_y = int(original_point[1] * scale_y)

        return (normalized_x, normalized_y)

    def _calculate_route_gas_cost(self, route: List[Tuple[float, float]]) -> float:
        """Calcula o custo da rota baseado no custo do combustível pela distância"""
        if len(route) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += calculate_distance(route[i], route[i + 1])

        total_distance_km = calculate_distance_in_km(total_distance, self.pixels_to_km_factor)

        gas_cost_per_km = total_distance_km * self.gas_cost_per_km
        return gas_cost_per_km * self.vehicle_quantity

    def united_fitness(
        self,
        path: List[Tuple[float, float]],
        use_normalized: bool = False
    ) -> float:
        """
            Calcula o fitness de forma unificada com o fitness original
            Feito para nao precisarmos mudar a função de fitness original
        """
        original_fitness = calculate_fitness(path)
        fitness_with_tax, _ = self.calculate_fitness_with_route_cost(
            path, original_fitness, use_normalized)
        gas_cost = self._calculate_route_gas_cost(path)

        return fitness_with_tax + gas_cost


    def calculate_fitness_with_route_cost(self, path: List[Tuple[float, float]], 
                                        original_fitness: float,
                                        use_normalized: bool = False) -> float:
        """
        Calcula fitness usando custos de rotas diretamente do city_route_cost.
        
        Args:
            path: Lista de coordenadas das cidades na solução
            original_fitness: Fitness original (distância)
            use_normalized: Se True, normaliza as coordenadas antes de buscar
            
        Returns:
            float: Fitness total = fitness_original + custo_total_rotas
        """
        total_route_cost = 0

        for i, current_city in enumerate(path):  # i é o índice, current_city é a cidade
            next_city = path[(i + 1) % len(path)]  # Próxima cidade usando o índice

            # Normaliza coordenadas se necessário
            if use_normalized:
                current_city = self.normalize_coordinates(current_city)
                next_city = self.normalize_coordinates(next_city)

            # Busca custo da rota no dicionário
            route_key = (current_city, next_city)
            reverse_key = (next_city, current_city)

            if route_key in self.city_route_cost:
                total_route_cost += self.city_route_cost[route_key]
            elif reverse_key in self.city_route_cost:
                total_route_cost += self.city_route_cost[reverse_key]
            # Se não encontrar, custo é 0 (rota gratuita)

        return original_fitness + total_route_cost, total_route_cost

    def united_fitness_with_route_cost(
        self,
        path: List[Tuple[float, float]],
        use_normalized: bool = False
    ) -> float:
        """
            Calcula o fitness de forma unificada com o fitness original
            Feito para nao precisarmos mudar a função de fitness original
            no momento já que precisamos colocar vários parametros.
        """
        original_fitness = calculate_fitness(path)
        return self.calculate_fitness_with_route_cost(path, original_fitness, use_normalized)

    def get_route_cost(self, city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
        """
        Obtém o custo de uma rota específica.
        
        Args:
            city1: Coordenadas da cidade origem
            city2: Coordenadas da cidade destino
            
        Returns:
            float: Custo da rota (0 se não especificado)
        """
        route_key = (city1, city2)
        reverse_key = (city2, city1)

        if route_key in self.city_route_cost:
            return self.city_route_cost[route_key]
        if reverse_key in self.city_route_cost:
            return self.city_route_cost[reverse_key]
        return 0.0

    def add_route_cost(self, city1: Tuple[float, float], city2: Tuple[float, float], cost: float):
        """
        Adiciona ou atualiza o custo de uma rota específica.
        
        Args:
            city1: Coordenadas da cidade origem
            city2: Coordenadas da cidade destino
            cost: Custo da rota
        """
        route_key = (city1, city2)
        self.city_route_cost[route_key] = cost

    def remove_route_cost(self, city1: Tuple[float, float], city2: Tuple[float, float]):
        """
        Remove o custo de uma rota específica.
        
        Args:
            city1: Coordenadas da cidade origem
            city2: Coordenadas da cidade destino
        """
        route_key = (city1, city2)
        reverse_key = (city2, city1)

        if route_key in self.city_route_cost:
            del self.city_route_cost[route_key]
        elif reverse_key in self.city_route_cost:
            del self.city_route_cost[reverse_key]

    def print_route_costs(self):
        """
        Imprime todos os custos de rotas definidos.
        """
        print("Custos de Rotas Definidos:")
        print("=" * 40)

        for (city1, city2), cost in self.city_route_cost.items():
            print(f"Rota {city1} -> {city2}: {cost}")

        print(f"Total de rotas com custo: {len(self.city_route_cost)}")

    def get_total_routes_with_cost(self) -> int:
        """
        Retorna o número total de rotas com custo definido.
        
        Returns:
            int: Número de rotas com custo
        """
        return len(self.city_route_cost)

    def get_route_description(self, path, use_normalized: bool = False) -> str:
        """
            Gera o relatório com o custo de combustível e pedágio
        """
        original_fitness = calculate_fitness(path)
        total_fitness, tax_cost = self.calculate_fitness_with_route_cost(
            path, original_fitness, use_normalized)
        gas_cost = self._calculate_route_gas_cost(path)

        return f"""
            O Custo final do melhor trajeto foi:

            Taxa de Fitness total: {round(total_fitness, 2)}
            Taxa de pedágio: R$ {round(tax_cost, 2)}
            Taxa de combustível: {round(gas_cost, 2)}
        """
