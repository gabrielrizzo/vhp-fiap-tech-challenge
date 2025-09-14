# Restrição: Múltiplas Ambulâncias 

## Descrição
Esta restrição implementa a funcionalidade de múltiplas ambulâncias que dividem os atendimentos entre diferentes pacientes espalhados pela cidade, otimizando tanto o tempo total de atendimento quanto a quantidade de veículos necessários.

### Contexto do Problema
- **Múltiplas ambulâncias** disponíveis para atendimento
- **Pacientes distribuídos** em diferentes localidades da cidade
- **Depósito central** (hospital base) de onde partem e retornam as ambulâncias
- **Otimização multi-objetivo**: tempo total + número de veículos + utilização

## Como Funciona

O sistema de otimização é implementado através da classe `MultipleVehicles` que:

1. **Distribui pacientes** entre ambulâncias usando algoritmo de clustering
2. **Otimiza rotas individuais** usando algoritmo do vizinho mais próximo
3. **Calcula métricas de performance** (tempo, utilização, eficiência)
4. **Integra com algoritmo genético** para otimização global

### Algoritmos Utilizados
- **Clustering simples** para distribuição inicial de pacientes
- **Algoritmo do vizinho mais próximo** para otimização de rotas
- **Função de fitness multi-objetivo** para balanceamento de critérios

## Funcionalidades Principais

### Classe MultipleVehicles
```python
class MultipleVehicles:
    def __init__(self, max_vehicles: int = 5, vehicle_capacity: int = 10)
    def assign_patients_to_vehicles(self, patients, depot)
    def get_total_time(self) -> float
    def get_average_time(self) -> float
    def get_vehicle_count(self) -> int
    def get_vehicle_utilization(self) -> Dict[int, float]
```

### Funções de Otimização
- `calculate_fitness_with_multiple_vehicles()` - Calcula fitness multi-objetivo
- `optimize_vehicle_assignment()` - Otimiza atribuição usando busca local

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
print(f"Tempo total: {metrics['total_time']:.2f} minutos")
print(f"Ambulâncias utilizadas: {metrics['vehicle_count']}")
print(f"Utilização média: {sum(metrics['vehicle_utilization'].values()) / metrics['vehicle_count']:.1%}")

# Rotas detalhadas
for vehicle_id, route in metrics['vehicle_routes'].items():
    print(f"Ambulância {vehicle_id}: {route}")
```

## Métricas Retornadas

- **Tempo total**: Tempo da ambulância mais lenta (em minutos)
- **Tempo médio**: Tempo médio de todas as ambulâncias
- **Número de veículos**: Quantidade de ambulâncias utilizadas
- **Utilização**: Percentual de ocupação de cada ambulância (0-1)
- **Rotas detalhadas**: Sequência completa de cada ambulância

## Exemplo Completo

Veja o arquivo `demo_multiple_vehicles.py` para um exemplo completo que demonstra:
- **Integração com algoritmo genético** para otimização de rotas
- **Distribuição automática** de pacientes entre múltiplas ambulâncias
- **Análise de eficiência** e recomendações automáticas
- **Visualização detalhada** das rotas e métricas

## Parâmetros Configuráveis

- `max_vehicles`: Número máximo de ambulâncias disponíveis
- `vehicle_capacity`: Capacidade máxima de pacientes por ambulância
- `time_weight`: Peso para otimização do tempo (padrão: 1.0)
- `vehicle_count_weight`: Peso para minimizar número de veículos (padrão: 10.0)
- `utilization_weight`: Peso para maximizar utilização (padrão: 5.0)
