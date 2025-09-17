# Interface de Restrições

Este documento descreve a interface padrão para implementação de restrições no problema do caixeiro viajante.

## Objetivo

A interface `RestrictionInterface` define um contrato que todas as restrições devem seguir para garantir compatibilidade com o algoritmo genético principal. Isso permite:

- Padronização da implementação de restrições
- Facilidade na adição de novas restrições
- Integração simplificada com a interface gráfica
- Composição de múltiplas restrições

## Métodos Obrigatórios

Toda classe que implementa a interface `RestrictionInterface` deve fornecer os seguintes métodos:

### `apply_restriction(path)`

```python
def apply_restriction(self, path: List[Tuple[float, float]]) -> float:
```

- **Descrição**: Aplica a restrição ao caminho fornecido e calcula uma penalidade.
- **Parâmetros**: 
  - `path`: Lista de coordenadas das cidades no caminho
- **Retorno**: Valor da penalidade a ser adicionada ao fitness

### `is_valid(path)`

```python
def is_valid(self, path: List[Tuple[float, float]]) -> bool:
```

- **Descrição**: Verifica se o caminho é válido de acordo com a restrição.
- **Parâmetros**:
  - `path`: Lista de coordenadas das cidades no caminho
- **Retorno**: `True` se o caminho é válido, `False` caso contrário

### `get_name()`

```python
def get_name(self) -> str:
```

- **Descrição**: Retorna o nome da restrição.
- **Retorno**: Nome da restrição

### `get_description()`

```python
def get_description(self) -> str:
```

- **Descrição**: Retorna a descrição detalhada da restrição.
- **Retorno**: Descrição da restrição

### `get_parameters()`

```python
def get_parameters(self) -> Dict[str, Any]:
```

- **Descrição**: Retorna os parâmetros configuráveis da restrição.
- **Retorno**: Dicionário com os parâmetros e seus valores atuais

### `set_parameters(parameters)`

```python
def set_parameters(self, parameters: Dict[str, Any]) -> None:
```

- **Descrição**: Configura os parâmetros da restrição.
- **Parâmetros**:
  - `parameters`: Dicionário com os parâmetros a serem configurados

## Como Implementar uma Nova Restrição

Para criar uma nova restrição, siga os passos abaixo:

1. Crie uma nova classe que herda de `RestrictionInterface`:

```python
from restrictions.restriction_interface import RestrictionInterface

class MinhaRestricao(RestrictionInterface):
    def __init__(self):
        # Inicialização da restrição
        pass
```

2. Implemente todos os métodos obrigatórios:

```python
def apply_restriction(self, path):
    # Lógica para calcular a penalidade
    return penalidade

def is_valid(self, path):
    # Lógica para verificar se o caminho é válido
    return True ou False

def get_name(self):
    return "Nome da Restrição"

def get_description(self):
    return "Descrição detalhada da restrição"

def get_parameters(self):
    return {"parametro1": valor1, "parametro2": valor2}

def set_parameters(self, parameters):
    if "parametro1" in parameters:
        self._parametro1 = parameters["parametro1"]
```

3. Implemente métodos adicionais específicos da sua restrição, se necessário.

## Exemplo: ForbiddenRoutes

A restrição `ForbiddenRoutes` é um exemplo de implementação da interface:

```python
class ForbiddenRoutes(RestrictionInterface):
    def __init__(self, base_distance_penalty=1000.0):
        self._forbidden_routes = set()
        self._base_distance_penalty = base_distance_penalty
        
    def apply_restriction(self, path):
        # Calcula penalidade para rotas proibidas
        # ...
        
    # Implementação dos outros métodos obrigatórios
    # ...
    
    # Métodos específicos desta restrição
    def add_forbidden_route(self, city1, city2):
        # ...
        
    def is_route_forbidden(self, city1, city2):
        # ...
```

## Integração com o Algoritmo Genético

Para usar uma restrição no algoritmo genético, basta:

1. Instanciar a restrição
2. Configurar seus parâmetros
3. Passar para a função de fitness

```python
# Exemplo de uso
restricao = MinhaRestricao()
restricao.set_parameters({"parametro1": 100})

# No cálculo do fitness
fitness = calculate_fitness(path) + restricao.apply_restriction(path)
```

## Integração com a Interface Gráfica

A interface gráfica pode usar os métodos `get_parameters()` e `set_parameters()` para permitir que o usuário configure as restrições dinamicamente.
