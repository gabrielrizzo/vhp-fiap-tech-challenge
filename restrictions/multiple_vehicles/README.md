# Restrição: Múltiplas Ambulâncias 

## Descrição
Esta restrição implementa a funcionalidade de múltiplas ambulâncias que dividem os atendimentos entre diferentes pacientes espalhados pela cidade, otimizando simultaneamente o tempo total de atendimento, a quantidade de veículos necessários e a utilização eficiente das ambulâncias.

### Contexto do Problema
- **Múltiplas ambulâncias** disponíveis para atendimento simultâneo
- **Pacientes distribuídos** em diferentes localidades da cidade
- **Depósito central** (hospital base) de onde partem e retornam as ambulâncias
- **Otimização multi-objetivo**: minimizar tempo total, minimizar número de veículos e maximizar utilização
- **Capacidade limitada** de cada ambulância (configurável)

## Como Funciona

O sistema de otimização é implementado através da classe `MultipleVehicles` que:

1. **Calcula número necessário** de ambulâncias baseado na capacidade e número de pacientes
2. **Distribui pacientes** entre ambulâncias usando algoritmo de clustering sequencial
3. **Otimiza rotas individuais** usando algoritmo do vizinho mais próximo
4. **Calcula métricas de performance** (tempo total, tempo médio, utilização, eficiência)
5. **Integra com algoritmo genético** através da função de fitness multi-objetivo

### Algoritmos Utilizados
- **Distribuição sequencial (clustering sequencial)** para atribuição inicial de pacientes às ambulâncias
- **Algoritmo do vizinho mais próximo** para otimização de rotas individuais
- **Função de fitness multi-objetivo** com pesos configuráveis para balanceamento de critérios
- **Cálculo de distância euclidiana** para otimização de rotas

## Funcionalidades Principais

### Classe MultipleVehicles
```python
class MultipleVehicles:
```

### Funções de Otimização
- `calculate_fitness_with_multiple_vehicles()` - Calcula fitness multi-objetivo com pesos configuráveis

## Como Usar

### 1. Configuração Básica
```python
from restrictions.multiple_vehicles import MultipleVehicles

# Cria sistema com 5 ambulâncias, capacidade de 8 pacientes cada
multiple_vehicles = MultipleVehicles(max_vehicles=5, vehicle_capacity=8)
```

### 2. Distribuição de Pacientes
```python
# Lista de coordenadas dos pacientes
patients = [(x1, y1), (x2, y2), ..., (xn, yn)]
depot = (50, 50)  # Coordenadas do hospital base

# Distribui pacientes entre ambulâncias
routes = multiple_vehicles.assign_patients_to_vehicles(patients, depot)
```

### 3. Análise de Resultados
```python
# Métricas disponíveis
print(f"Tempo total: {multiple_vehicles.get_total_time():.2f} minutos")
print(f"Tempo médio: {multiple_vehicles.get_average_time():.2f} minutos")
print(f"Ambulâncias utilizadas: {multiple_vehicles.get_vehicle_count()}")
print(f"Utilização média: {sum(multiple_vehicles.get_vehicle_utilization().values()) / multiple_vehicles.get_vehicle_count():.1%}")

# Rotas detalhadas
for vehicle_id, route in multiple_vehicles.vehicle_routes.items():
    print(f"Ambulância {vehicle_id}: {route}")
```

## Métricas Retornadas

- **Tempo total**: Tempo da ambulância mais lenta (em minutos) - critério de gargalo
- **Tempo médio**: Tempo médio de todas as ambulâncias (em minutos)
- **Número de veículos**: Quantidade de ambulâncias utilizadas
- **Utilização**: Percentual de ocupação de cada ambulância (0-1), onde 1.0 = 100% da capacidade
- **Rotas detalhadas**: Sequência completa de cada ambulância (depósito ou hospital base → pacientes → depósito)
- **Distância total**: Soma de todas as distâncias percorridas por todas as ambulâncias

## Exemplo Completo

Veja o arquivo `demo_multiple_vehicles.py` para um exemplo completo que demonstra:
- **Integração com algoritmo genético** para otimização de rotas
- **Distribuição automática** de pacientes entre múltiplas ambulâncias
- **Análise de eficiência** e recomendações automáticas
- **Visualização detalhada** das rotas e métricas

## Parâmetros Configuráveis

### Parâmetros da Classe MultipleVehicles
- `max_vehicles`: Número máximo de ambulâncias disponíveis (padrão: 5)
- `vehicle_capacity`: Capacidade máxima de pacientes por ambulância (padrão: 10)

### Parâmetros da Função de Fitness
- `time_weight`: Peso para otimização do tempo total (padrão: 1.0)
  - Controla a importância de minimizar o tempo da ambulância mais lenta
  - Valores maiores priorizam soluções mais rápidas
- `vehicle_count_weight`: Peso para minimizar número de veículos (padrão: 10.0)
  - Penaliza o uso excessivo de ambulâncias para otimizar recursos
  - Valores maiores incentivam soluções com menos veículos
- `utilization_weight`: Peso para maximizar utilização dos veículos (padrão: 5.0)
  - Recompensa o uso eficiente da capacidade das ambulâncias
  - Valores maiores priorizam ambulâncias com maior ocupação

### Fórmula do Fitness
```
fitness = (time_weight × tempo_total) + (vehicle_count_weight × n_veículos) - (utilization_weight × utilização_média)
```

**Nota**: Menor valor de fitness = melhor solução

## Características Técnicas

### Velocidade e Tempo
- **Velocidade padrão**: 50 km/h (configurável no método `_calculate_route_time`)
- **Unidade de tempo**: Minutos
- **Cálculo de distância**: Distância euclidiana entre coordenadas

### Algoritmo de Distribuição
- **Estratégia**: Distribuição sequencial balanceada
- **Balanceamento**: Distribui pacientes restantes entre as primeiras ambulâncias
- **Otimização**: Cada rota individual é otimizada com algoritmo do vizinho mais próximo

### Limitações
- **Capacidade fixa**: Todas as ambulâncias têm a mesma capacidade
- **Velocidade constante**: Não considera tráfego ou condições de estrada
- **Distribuição simples**: Não usa clustering geográfico avançado
