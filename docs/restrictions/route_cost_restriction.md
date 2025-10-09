# Restrição: Custo de Rotas

## Descrição
Esta restrição implementa o controle de custos adicionais nas rotas do problema do caixeiro viajante, simulando situações reais como:
- Pedágios em rodovias
- Taxas de travessia (pontes, túneis, balsas)
- Custos diferenciados por tipo de via
- Tarifas urbanas (zona azul, congestionamento)

## Como Funciona

A restrição é implementada através da classe `RouteCostRestriction` que:
1. Mantém um dicionário com custos específicos para cada rota
2. Calcula o custo total somando os valores de cada trecho
3. Adiciona essa informação ao fitness da solução
4. Permite que rotas sem custo definido sejam consideradas gratuitas

### Cálculo de Custo
Para cada trecho da rota:
- Busca o custo no dicionário usando o par de cidades (origem, destino)
- Considera rotas bidirecionais automaticamente
- Se não houver custo definido, assume custo zero (rota gratuita)
- Soma todos os custos para obter o total da rota

## Como Usar

1. Prepare suas cidades e dicionário de custos:
```python
# Lista de localizações das cidades
cities = [
    (100, 100),  # Cidade A
    (200, 150),  # Cidade B
    (300, 100),  # Cidade C
    (250, 250)   # Cidade D
]

# Dicionário de custos (apenas rotas com custo)
route_costs = {
    ((100, 100), (200, 150)): 15.50,  # Pedágio A -> B
    ((200, 150), (300, 100)): 22.00,  # Pedágio B -> C
    ((300, 100), (250, 250)): 8.75,   # Taxa C -> D
}
```

2. Crie o objeto `RouteCostRestriction`:
```python
from restrictions.route_cost_restriction import RouteCostRestriction

route_cost = RouteCostRestriction(
    cities_locations=cities,
    route_cost_dict=route_costs,
    weight=1.0  # peso da restrição no fitness
)
```

3. Calcule o custo de uma rota:
```python
# Rota de exemplo
route = [(100, 100), (200, 150), (300, 100), (250, 250), (100, 100)]

# Calcular custo total
penalty = route_cost.calculate_penalty(route)
print(f"Custo total: R$ {penalty:.2f}")
```

4. Obtenha informações detalhadas:
```python
# Retorna métricas sobre o custo
cost_info = route_cost.get_route_cost(route)
print(f"Custo de rotas: R$ {cost_info['total_route_cost']:.2f}")
```

## Exemplo Completo

```python
from restrictions.route_cost_restriction import RouteCostRestriction

# Definir cidades
cities = [
    (0, 0),      # Centro
    (100, 0),    # Leste
    (100, 100),  # Nordeste
    (0, 100)     # Norte
]

# Definir custos (apenas rotas com pedágio/taxa)
costs = {
    ((0, 0), (100, 0)): 12.50,      # Pedágio na via expressa
    ((100, 0), (100, 100)): 8.00,   # Taxa de ponte
    ((100, 100), (0, 100)): 15.00,  # Pedágio + taxa
    # Rota (0, 100) -> (0, 0) não tem custo (gratuita)
}

# Criar restrição
route_cost = RouteCostRestriction(
    cities_locations=cities,
    route_cost_dict=costs,
    weight=1.5  # dar mais peso aos custos
)

# Testar rota
route = [(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)]

# Validar (sempre retorna True)
if route_cost.validate_route(route):
    cost_info = route_cost.get_route_cost(route)
    total = cost_info['total_route_cost']
    print(f"Custo total da rota: R$ {total:.2f}")
    
    if total > 30:
        print("⚠️ Rota com custo elevado, considere alternativas")
```

## Parâmetros Configuráveis

- `cities_locations`: Lista de tuplas com coordenadas (x, y) de cada cidade
  - Representa todas as cidades do problema
  - Usado para referência e validação

- `route_cost_dict`: Dicionário com custos de rotas específicas
  - Chave: tupla de tuplas `((x1, y1), (x2, y2))` representando origem e destino
  - Valor: custo numérico da rota (float)
  - Rotas não incluídas são consideradas gratuitas

- `weight` (padrão: 1.0): Peso da restrição no cálculo do fitness
  - Valores maiores aumentam a importância dos custos
  - Valores menores permitem que outros fatores predominem

## Características Importantes

### Rotas Bidirecionais
A restrição automaticamente considera rotas bidirecionais:
```python
# Estas duas definições são equivalentes:
costs = {
    ((A, B)): 10.0  # A -> B custa 10
}
# O sistema também aplicará custo 10 para B -> A
```

### Rotas Gratuitas
Rotas não definidas no dicionário têm custo zero:
```python
costs = {
    ((A, B)): 15.0  # A -> B tem custo
    # C -> D não está definido, logo custo = 0
}
```

### Flexibilidade de Aplicação
A restrição pode representar diversos tipos de custo:
- **Pedágios**: Custos fixos por trecho
- **Taxas variáveis**: Valores diferentes por horário/dia
- **Custos operacionais**: Desgaste, manutenção específica
- **Tarifas urbanas**: Zona azul, rodízio, congestionamento

## Métodos Disponíveis

### `validate_route(route, vehicle_data=None)`
Sempre retorna `True`, pois esta restrição não invalida rotas, apenas adiciona custos.

### `calculate_penalty(route, vehicle_data=None)`
Calcula e retorna o custo total da rota somando os custos de cada trecho.

### `get_route_cost(route)`
Retorna um dicionário com:
- `total_route_cost`: Custo total acumulado da rota

## Limitações

1. **Custos estáticos**
   - Os custos são fixos durante toda a execução
   - Não considera variações por horário ou dia da semana
   - Para custos dinâmicos, seria necessário estender a classe

2. **Sem validação de orçamento**
   - A restrição não impede rotas caras, apenas calcula o custo
   - Para limitar orçamento, combine com outras restrições

3. **Custos uniformes bidirecionais**
   - O custo A -> B é igual ao custo B -> A
   - Para custos direcionais diferentes, use a restrição de rotas unidirecionais

4. **Sem custos compostos**
   - Não calcula automaticamente custos baseados em distância
   - Para isso, use a restrição de combustível combinada

## Integração com Outras Restrições

Esta restrição funciona bem combinada com:
- **Combustível**: Custo total = combustível + pedágios
- **Rotas proibidas**: Evitar rotas caras e bloqueadas
- **Janelas de tempo**: Considerar custos vs. prazos
- **Rotas unidirecionais**: Custos diferentes por direção

## Exemplo de Integração

```python
from restrictions.route_cost_restriction import RouteCostRestriction
from restrictions.fuel_restriction import FuelRestriction

# Criar múltiplas restrições
route_cost = RouteCostRestriction(cities, costs, weight=1.0)
fuel = FuelRestriction(max_distance=300, fuel_cost_per_km=0.8)

# Calcular custo total
def total_cost(route):
    toll_cost = route_cost.calculate_penalty(route)
    fuel_cost = fuel.get_fuel_consumption(route)['fuel_cost']
    return toll_cost + fuel_cost

# Avaliar rota
route = [(0, 0), (100, 100), (200, 50), (0, 0)]
total = total_cost(route)
print(f"Custo total (pedágios + combustível): R$ {total:.2f}")
```

## Dicas de Uso

1. **Definição de custos**:
   - Inclua apenas rotas com custo real
   - Mantenha o dicionário organizado e documentado
   - Use valores realistas baseados em dados reais

2. **Ajuste de peso**:
   - Use `weight=1.0` para custos normais
   - Aumente para priorizar economia
   - Reduza se custos forem menos importantes que tempo/distância

3. **Manutenção**:
   - Atualize os custos periodicamente
   - Documente a fonte dos valores
   - Considere inflação e variações sazonais

4. **Performance**:
   - O dicionário permite buscas O(1) por rota
   - Eficiente mesmo com milhares de rotas
   - Sem overhead computacional significativo