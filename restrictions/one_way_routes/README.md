# Restrição: Rotas Unidirecionais

## Descrição
Esta restrição implementa a funcionalidade de rotas unidirecionais (vias de mão única) no problema do caixeiro viajante, simulando situações reais como:
- Ruas de mão única
- Vias com sentido único de circulação
- Trechos com restrição de direção
- Fluxos de tráfego controlados

## Como Funciona

A restrição é implementada através da classe `OneWayRoutes` que:
1. Mantém um registro de rotas unidirecionais entre pares de cidades
2. Penaliza soluções que tentam usar essas rotas na direção contrária (contramão)
3. Guia o algoritmo genético para encontrar caminhos que respeitem o sentido das vias

### Penalização
Quando uma solução tenta usar uma rota na contramão:
- Uma penalidade é adicionada ao fitness da solução
- A penalidade é proporcional à distância da rota em contramão
- Isso faz com que o algoritmo naturalmente evite ir na contramão

## Como Usar

1. Crie um objeto `OneWayRoutes`:
```python
from restrictions.one_way_routes import OneWayRoutes
one_way_routes = OneWayRoutes()
```

2. Adicione as rotas unidirecionais (definindo a direção permitida):
```python
# Permite apenas a direção cidade_a -> cidade_b
# (A direção cidade_b -> cidade_a será considerada contramão)
one_way_routes.add_one_way_route(cidade_a, cidade_b)
```

3. Use o objeto nas funções de fitness:
```python
# O fitness será calculado considerando penalidades para rotas na contramão
fitness = calculate_fitness(solucao, one_way_routes)
```

## Exemplo Completo

Veja o arquivo `demo_one_way_routes.py` para um exemplo completo que demonstra:
- Como criar e configurar rotas unidirecionais
- Como integrar com o algoritmo genético
- Como verificar se a solução final respeita as restrições de direção

## Parâmetros Configuráveis

- `base_distance_penalty` (padrão: 2000.0): Penalidade base para cada rota em contramão
  - Valores maiores tornam as restrições mais "rígidas"
  - Valores menores permitem que o algoritmo ocasionalmente viole as restrições se o benefício for grande

## Métodos Principais

- `add_one_way_route(origin, destination)`: Adiciona uma rota unidirecional permitida apenas na direção origin → destination
- `is_wrong_way(origin, destination)`: Verifica se uma rota está na contramão
- `get_all_one_way_routes()`: Retorna todas as rotas unidirecionais definidas
- `clear_one_way_routes()`: Remove todas as rotas unidirecionais

## Implementação da Interface

Esta restrição implementa a interface `RestrictionInterface`, fornecendo:

- `fitness_restriction(path)`: Calcula a penalidade para um caminho
- `is_valid(path)`: Verifica se o caminho não contém rotas na contramão
- `get_name()`: Retorna o nome da restrição
- `get_description()`: Retorna a descrição da restrição
- `get_parameters()`: Retorna os parâmetros configuráveis
- `set_parameters(parameters)`: Configura os parâmetros

## Integração com Outras Restrições

Esta restrição pode ser combinada com outras, como:
- Rotas proibidas
- Custos diferenciados por rota
- Janelas de tempo
- Distância máxima total
