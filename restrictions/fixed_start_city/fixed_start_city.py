from typing import List, Tuple, Protocol


class RestrictionInterface(Protocol):
    """Interface para restrições do problema do caixeiro viajante."""

    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """Verifica se o caminho é válido para a restrição."""
        pass

    def fitness_restriction(self, path: List[Tuple[float, float]]) -> float:
        """Calcula a penalidade se o caminho não respeitar a restrição."""
        pass

    def get_description(self) -> str:
        """Retorna a descrição da restrição."""
        pass


class FixedStartCity:
    """Implementa a restrição de cidade inicial fixa para o TSP."""

    def __init__(self):
        """Inicializa a restrição de cidade inicial fixa."""
        self.start_city: Tuple[float, float] = None  # type: ignore

    def set_start_city(self, city: Tuple[float, float]) -> None:
        """Define a cidade inicial obrigatória."""
        self.start_city = city

    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """Verifica se o caminho começa na cidade inicial definida."""
        if not path or self.start_city is None:
            # Se não há caminho ou cidade inicial definida, é válido
            return True
        return path[0] == self.start_city

    def fitness_restriction(self, path: List[Tuple[float, float]]) -> float:
        """Calcula a penalidade se o caminho não começar na cidade inicial."""
        if not self.is_valid(path):
            # Penalidade máxima se não começar na cidade correta
            return float('inf')
        return 0.0

    def get_description(self) -> str:
        """Retorna a descrição da restrição."""
        if self.start_city is None:
            return "Restrição de cidade inicial fixa (não definida)"
        return f"Restrição de cidade inicial fixa: início em {self.start_city}"