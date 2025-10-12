# Restrição: Múltiplos Veículos (Frota)

## Descrição
Esta restrição implementa o controle de frota de múltiplos veículos no problema do caixeiro viajante médico, simulando situações reais como:
- Distribuição de pacientes entre ambulâncias (integração com a restrição de capacidade do veículo)
- Otimização de rotas para múltiplas ambulâncias

## Como Funciona

A restrição é implementada através da classe `MultipleVehiclesRestriction` que:
1. Distribui pacientes entre veículos respeitando a capacidade máxima por veículo
2. Cria rotas otimizadas para cada ambulância usando algoritmo do vizinho mais próximo
3. Calcula métricas de tempo, utilização e eficiência da frota
4. Penaliza soluções que excedem o número máximo de veículos disponíveis

### Distribuição de Pacientes
O sistema distribui pacientes sequencialmente entre veículos:
- Cada veículo tem capacidade limitada (`vehicle_capacity`)
- Pacientes são atribuídos respeitando a ordem da rota
- Cada veículo cria sua própria rota: depósito → pacientes → depósito
- Algoritmo do vizinho mais próximo otimiza cada rota individual
- *Caso a capacidade do veículo não seja informada, é realizado um calculo para definir uma capacidade ótima para cada veículo vehicle_capacity = -(-qtd_cidades // max_vehicles)*

### Penalização
Quando uma solução excede os limites da frota:
- **Excesso de veículos**: Penalidade de 100 pontos por veículo adicional necessário
- **Tempo total**: Penalidade de 0.1 pontos por minuto de tempo total
- **Pacientes não atendidos**: Penalidade de 500 pontos por paciente sem atendimento
- **Bônus de utilização**: Redução de até 50 pontos por boa utilização média

## Como Usar

1. Crie um objeto `MultipleVehiclesRestriction`:
```python
from restrictions.multiple_vehicles import MultipleVehiclesRestriction

# Configuração básica
multiple_vehicles = MultipleVehiclesRestriction(
    max_vehicles=5,                    # máximo de ambulâncias
    depot=(0, 0),                      # coordenadas do hospital base
    vehicle_capacity=3                 # pacientes por ambulância
)
```

2. Configure parâmetros específicos:
```python
# Definir hospital base
multiple_vehicles.set_depot((100, 100))

# Ajustar capacidade por veículo
multiple_vehicles.set_vehicle_capacity(4)
```

3. Valide uma rota:
```python
# Verifica se a rota pode ser atendida com a frota disponível
is_valid = multiple_vehicles.validate_route(route)
```

4. Calcule penalidades:
```python
# Obtém a penalidade para rotas que excedem a capacidade da frota
penalty = multiple_vehicles.calculate_penalty(route)
```

5. Obtenha métricas da frota:
```python
# Retorna informações detalhadas sobre a distribuição
info = multiple_vehicles.get_multiple_vehicles_info(route)
print(f"Veículos necessários: {info['vehicles_needed']}")
print(f"Tempo total: {info['total_time']:.1f} min")
print(f"Utilização média: {info['average_utilization']:.1%}")
```

## Exemplo Completo

```python
from restrictions.multiple_vehicles import MultipleVehiclesRestriction

# Criar restrição de frota
fleet = MultipleVehiclesRestriction(
    max_vehicles=3,
    depot=(50, 50),           # Hospital central
    vehicle_capacity=2        # 2 pacientes por ambulância
)

# Rota com 7 pacientes
route = [
    (50, 50),   # Hospital (depósito)
    (10, 10),   # Paciente 1
    (20, 20),   # Paciente 2
    (30, 30),   # Paciente 3
    (40, 40),   # Paciente 4
    (60, 60),   # Paciente 5
    (70, 70),   # Paciente 6
    (80, 80),   # Paciente 7
    (50, 50)    # Retorno ao hospital
]

# Validar rota
if fleet.validate_route(route):
    print("✅ Rota pode ser atendida com a frota disponível")
    
    # Obter informações detalhadas
    info = fleet.get_multiple_vehicles_info(route)
    print(f"📊 Veículos necessários: {info['vehicles_needed']}")
    print(f"🚑 Veículos disponíveis: {info['max_vehicles']}")
    print(f"⏱️ Tempo total: {info['total_time']:.1f} min")
    print(f"📈 Utilização média: {info['average_utilization']:.1%}")
    
    # Mostrar utilização por veículo
    for vehicle_id, utilization in info['vehicle_utilization'].items():
        print(f"   Ambulância {vehicle_id}: {utilization:.1%}")
else:
    print("❌ Rota excede a capacidade da frota")
    penalty = fleet.calculate_penalty(route)
    print(f"Penalidade: {penalty:.1f}")
```

## Parâmetros Configuráveis

- `max_vehicles` (padrão: 5): Número máximo de ambulâncias disponíveis
  - Representa o limite da frota hospitalar
  - Rotas que exigem mais veículos recebem penalização alta
  - Deve ser ajustado conforme recursos disponíveis

- `depot` (padrão: None): Coordenadas do depósito/hospital base
  - Ponto de partida e retorno de todas as ambulâncias
  - Se não definido, a restrição não é aplicada
  - Deve ser uma das cidades da rota

- `vehicle_capacity` (padrão: 1): Capacidade de pacientes por ambulância
  - Limita quantos pacientes cada veículo pode transportar
  - Valores maiores permitem menos veículos para a mesma demanda
  - Deve refletir a capacidade real das ambulâncias

## Métricas Disponíveis

O método `get_multiple_vehicles_info()` retorna:
- `depot`: Coordenadas do hospital base
- `patients_count`: Número total de pacientes na rota
- `vehicles_needed`: Quantidade mínima de veículos necessários
- `max_vehicles`: Limite máximo da frota
- `can_handle_route`: Se a frota pode atender toda a demanda
- `total_time`: Tempo da ambulância mais lenta (min)
- `average_time`: Tempo médio das ambulâncias (min)
- `average_utilization`: Utilização média da frota (0-1)
- `vehicle_utilization`: Utilização individual por veículo
- `vehicle_capacity`: Capacidade configurada por veículo
- `unserved_patients`: Pacientes que não podem ser atendidos

## Métodos Principais

### Métricas de Tempo
- `get_total_time()`: Tempo da ambulância mais lenta
- `get_average_time()`: Tempo médio de todas as ambulâncias

### Métricas de Utilização
- `get_vehicle_count()`: Número de ambulâncias utilizadas
- `get_average_utilization()`: Utilização média da frota (0-1)
- `get_vehicle_utilization()`: Utilização individual por veículo

### Integração
- `get_vehicle_data_for_capacity_restriction()`: Dados para integração com outras restrições

## Integração com Outras Restrições

Esta restrição fornece dados para integração com:
- **VehicleCapacityRestriction**: Usa `vehicle_capacity`, `max_vehicles`, `vehicles_used`, `unserved_patients`
- **FuelRestriction**: Pode usar `vehicles_used` para calcular consumo total da frota
- **RouteCostRestriction**: Pode multiplicar custos pelo número de veículos

### Exemplo de Integração
```python
from restrictions.multiple_vehicles import MultipleVehiclesRestriction
from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction

# Criar restrições
fleet = MultipleVehiclesRestriction(max_vehicles=3, depot=(0,0), vehicle_capacity=2)
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=2)

# Obter dados de integração
vehicle_data = fleet.get_vehicle_data_for_capacity_restriction(route)

# Usar dados na validação de capacidade
is_valid = capacity.validate_route(route, vehicle_data)
penalty = capacity.calculate_penalty(route, vehicle_data)
```

## Limitações

1. **Distribuição sequencial**
   - Pacientes são distribuídos na ordem da rota
   - Não considera otimização global da distribuição
   - Para distribuição ótima, seria necessário algoritmo mais complexo

2. **Algoritmo simples de roteamento**
   - Usa apenas vizinho mais próximo para cada veículo
   - Não considera otimização entre veículos
   - Para rotas mais eficientes, seria necessário algoritmo de frota

3. **Sem consideração de janelas de tempo**
   - Não considera horários de atendimento dos pacientes
   - Para problemas com time windows, seria necessário extensão

4. **Capacidade uniforme**
   - Todos os veículos têm a mesma capacidade
   - Para frota heterogênea, seria necessário estender a classe

## Características Importantes

### Algoritmo de Distribuição
- **Sequencial**: Pacientes são atribuídos na ordem da rota
- **Respeitando capacidade**: Cada veículo não excede sua capacidade
- **Limitado pela frota**: Não usa mais veículos que o máximo disponível

### Cálculo de Rotas
- **Vizinho mais próximo**: Cada veículo usa algoritmo simples
- **Depósito obrigatório**: Todas as rotas começam e terminam no depósito
- **Otimização individual**: Cada veículo otimiza sua própria rota

### Métricas de Eficiência
- **Utilização**: Percentual de capacidade utilizada por veículo
- **Tempo total**: Tempo da ambulância mais lenta (gargalo)
- **Pacientes não atendidos**: Quantos ficam sem atendimento

## Dicas de Uso

1. **Configuração da frota**:
   - Ajuste `max_vehicles` conforme recursos disponíveis
   - Defina `vehicle_capacity` baseado na capacidade real das ambulâncias
   - Configure `depot` como o hospital principal

2. **Interpretação de métricas**:
   - `average_utilization` próximo de 1.0 indica boa eficiência
   - `total_time` alto pode indicar necessidade de mais veículos
   - `unserved_patients` > 0 indica capacidade insuficiente

3. **Integração com outras restrições**:
   - Use `get_vehicle_data_for_capacity_restriction()` para integração
   - Combine com restrições de combustível para custos totais
   - Considere restrições de tempo para otimização temporal

4. **Otimização**:
   - Aumente `vehicle_capacity` para reduzir número de veículos
   - Ajuste `max_vehicles` conforme demanda esperada
   - Monitore `average_utilization` para eficiência da frota
