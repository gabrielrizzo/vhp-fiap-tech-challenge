# Restrição: Combustível

## Descrição
Esta restrição implementa o controle de consumo de combustível no problema do caixeiro viajante, simulando situações reais como:
- Autonomia limitada do veículo
- Controle de custos operacionais
- Orçamento limitado de combustível
- Eficiência energética de rotas

## Como Funciona

A restrição é implementada através da classe `FuelRestriction` que:
1. Monitora a distância total percorrida pela rota
2. Calcula o custo de combustível baseado na distância
3. Penaliza soluções que excedem os limites estabelecidos
4. Fornece métricas de consumo e eficiência

### Penalização
Quando uma solução excede os limites:
- **Excesso de distância**: Penalidade de 10x o custo do combustível para cada km além do limite
- **Excesso de custo**: Penalidade de 3x o valor excedente do orçamento
- As penalidades são cumulativas quando ambos os limites são violados
- Isso guia o algoritmo a encontrar rotas economicamente viáveis

## Como Usar

1. Crie um objeto `FuelRestriction`:
```python
from restrictions.fuel_restriction import FuelRestriction

# Configuração básica (apenas distância máxima)
fuel_restriction = FuelRestriction(
    max_distance=250.0,  # km
    fuel_cost_per_km=0.8  # custo por km
)

# Configuração completa (com limite de custo)
fuel_restriction = FuelRestriction(
    max_distance=250.0,
    fuel_cost_per_km=0.8,
    fuel_cost_limit=200.0,  # orçamento máximo
    pixel_to_km_factor=0.02  # fator de conversão
)
```

2. Valide uma rota:
```python
# Verifica se a rota respeita os limites
is_valid = fuel_restriction.validate_route(route)
```

3. Calcule penalidades:
```python
# Obtém a penalidade para rotas inválidas
penalty = fuel_restriction.calculate_penalty(route)
```

4. Obtenha métricas de consumo:
```python
# Retorna informações detalhadas sobre o consumo
metrics = fuel_restriction.get_fuel_consumption(route)
print(f"Distância total: {metrics['total_distance']} km")
print(f"Custo de combustível: R$ {metrics['fuel_cost']}")
print(f"Capacidade restante: {metrics['remaining_fuel_capacity']} km")
print(f"Eficiência: {metrics['fuel_efficiency']:.2%}")
```

## Exemplo Completo

```python
from restrictions.fuel_restriction import FuelRestriction

# Criar restrição
fuel = FuelRestriction(
    max_distance=300.0,
    fuel_cost_per_km=0.75,
    fuel_cost_limit=250.0
)

# Rota de exemplo
route = [(0, 0), (10, 10), (20, 5), (30, 15), (0, 0)]

# Validar rota
if fuel.validate_route(route):
    print("Rota válida!")
    metrics = fuel.get_fuel_consumption(route)
    print(f"Consumo: R$ {metrics['fuel_cost']:.2f}")
else:
    print("Rota inválida!")
    penalty = fuel.calculate_penalty(route)
    print(f"Penalidade: {penalty:.2f}")
```

## Parâmetros Configuráveis

- `max_distance` (padrão: 250.0): Distância máxima em km que o veículo pode percorrer
  - Representa a autonomia do veículo
  - Rotas que excedem este valor recebem penalização

- `fuel_cost_per_km` (padrão: 0.8): Custo de combustível por quilômetro
  - Usado para calcular o custo total da rota
  - Permite simular diferentes tipos de veículos/combustíveis

- `fuel_cost_limit` (padrão: None): Orçamento máximo para combustível
  - Quando definido, restringe o custo total
  - Se None, apenas a distância é limitada

- `pixel_to_km_factor` (padrão: 0.02): Fator de conversão de pixels para quilômetros
  - Converte distâncias no mapa para unidades reais
  - Ajuste conforme a escala do seu problema

## Métricas Disponíveis

O método `get_fuel_consumption()` retorna:
- `total_distance`: Distância total percorrida (km)
- `fuel_cost`: Custo total de combustível (R$)
- `remaining_fuel_capacity`: Distância restante que pode ser percorrida (km)
- `fuel_efficiency`: Percentual de utilização da capacidade (0-1)

## Limitações

1. **Modelo de consumo linear**
   - Assume consumo constante por km
   - Não considera variações de terreno, trânsito ou velocidade
   - Para modelos mais complexos, seria necessário estender a classe

2. **Sem reabastecimento**
   - A restrição assume que não há pontos de reabastecimento
   - Para incluir postos de gasolina, seria necessário implementar uma nova restrição

3. **Custo único de combustível**
   - O custo por km é o mesmo para toda a rota
   - Não considera variações regionais de preço