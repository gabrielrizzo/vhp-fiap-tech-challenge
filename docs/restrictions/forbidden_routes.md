# Restrição: Rotas Proibidas

## Descrição
Esta restrição implementa a funcionalidade de rotas proibidas no problema do caixeiro viajante, simulando situações reais como:
- Ruas interditadas por enchentes
- Vias bloqueadas por obras
- Avenidas fechadas por eventos
- Trechos com restrição de circulação

## Como Funciona

A restrição é implementada através da classe `ForbiddenRoutes` que:
1. Mantém um registro de rotas proibidas entre pares de cidades
2. Penaliza soluções que tentam usar essas rotas
3. Guia o algoritmo genético para encontrar caminhos alternativos

### Penalização
Quando uma solução tenta usar uma rota proibida:
- Uma penalidade é adicionada ao fitness da solução
- A penalidade é proporcional à distância da rota proibida
- Isso faz com que o algoritmo naturalmente evite essas rotas

## Como Usar

1. Crie um objeto `ForbiddenRoutes`:
```python
from restrictions.forbidden_routes import ForbiddenRoutes
forbidden_routes = ForbiddenRoutes()
```

2. Adicione as rotas proibidas:
```python
# Proíbe a rota entre as cidades A e B
forbidden_routes.add_forbidden_route(cidade_a, cidade_b)
```

3. Use o objeto nas funções de fitness:
```python
# O fitness será calculado considerando penalidades para rotas proibidas
fitness = calculate_fitness(solucao, forbidden_routes)
```

## Exemplo Completo

Veja o arquivo `demo_forbidden_routes.py` para um exemplo completo que demonstra:
- Como criar e configurar rotas proibidas
- Como integrar com o algoritmo genético
- Como verificar se a solução final respeita as restrições

## Parâmetros Configuráveis

- `base_distance_penalty` (padrão: 1000.0): Penalidade base para cada rota proibida
  - Valores maiores tornam as restrições mais "rígidas"
  - Valores menores permitem que o algoritmo ocasionalmente viole as restrições se o benefício for grande

## Limitações

1. A restrição atual considera as rotas como bidirecionais
   - Se a rota A → B é proibida, B → A também será
   - Para rotas unidirecionais, será necessário implementar uma restrição separada

2. Penalidades fixas
   - A penalidade base é a mesma para todas as rotas
   - Futuramente pode ser interessante ter penalidades diferentes por rota

## Integração com Outras Restrições

Esta restrição pode ser combinada com outras, como:
- Rotas unidirecionais
- Custos diferenciados por rota
- Janelas de tempo
- Distância máxima total


[Texto do link](../one_way_routes/README.md)