from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any


class RestrictionInterface(ABC):
    """
    Interface base para todas as restrições aplicadas ao problema do caixeiro viajante.
    
    Todas as restrições devem implementar esta interface para garantir compatibilidade
    com o algoritmo genético principal.
    """
    
    @abstractmethod
    def fitness_restriction(self, path: List[Tuple[float, float]]) -> float:
        """
        Aplica a restrição ao caminho fornecido e retorna uma penalidade.
        
        Args:
            path: Lista de coordenadas das cidades no caminho
            
        Returns:
            float: Valor da penalidade a ser adicionada ao fitness
        """
        pass
 
    @abstractmethod
    def is_valid(self, path: List[Tuple[float, float]]) -> bool:
        """
        Verifica se o caminho é válido de acordo com a restrição.
        
        Args:
            path: Lista de coordenadas das cidades no caminho
            
        Returns:
            bool: True se o caminho é válido, False caso contrário
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Retorna o nome da restrição.
        
        Returns:
            str: Nome da restrição
        """
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """
        Retorna a descrição da restrição.
        
        Returns:
            str: Descrição detalhada da restrição
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Retorna os parâmetros configuráveis da restrição.
        
        Returns:
            Dict[str, Any]: Dicionário com os parâmetros e seus valores atuais
        """
        pass
    
    @abstractmethod
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Configura os parâmetros da restrição.
        
        Args:
            parameters: Dicionário com os parâmetros a serem configurados
        """
        pass

