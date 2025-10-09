# Restrição: Capacidade de Veículo

## Descrição
Esta restrição implementa o controle de capacidade por veículo no problema do caixeiro viajante médico, simulando situações reais como:
- Limite de pacientes por ambulância
- Controle de lotação máxima dos veículos

## Como Funciona

A restrição é implementada através da classe `VehicleCapacityRestriction` que:
1. Valida se o número de pacientes não excede a capacidade máxima por veículo
2. Avalia a capacidade total disponível
3. Calcula penalidades baseadas no excesso de pacientes
4. Fornece métricas de utilização e eficiência da capacidade

### Validação de Capacidade
O sistema verifica a capacidade em diferentes cenários:
- **Veículo único**: Valida contra capacidade de um veículo
- **Frota múltipla**: Considera capacidade total da frota disponível

### Penalização
Quando uma solução excede a capacidade:
- **Pacientes não atendidos**: Penalidade de 300 pontos por paciente sem atendimento
- **Excesso de capacidade**: Penalidade de 100 pontos por paciente acima da capacidade
- **Saturação alta**: Penalidade adicional quando utilização > 90%
- **Integração com frota**: Considera sinais de `unserved_patients` quando disponível

## Como Usar

1. Crie um objeto `VehicleCapacityRestriction`:
```python
from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction

# Configuração básica
capacity_restriction = VehicleCapacityRestriction(
    max_patients_per_vehicle=3  # máximo de pacientes por veículo
)
```

2. Valide uma rota:
```python
# Verifica se a rota respeita a capacidade
is_valid = capacity_restriction.validate_route(route)
```

3. Calcule penalidades:
```python
# Obtém a penalidade para rotas que excedem a capacidade
penalty = capacity_restriction.calculate_penalty(route)
```

4. Obtenha informações de capacidade:
```python
# Retorna métricas detalhadas sobre utilização
info = capacity_restriction.get_capacity_info(route)
print(f"Pacientes: {info['patient_count']}")
print(f"Capacidade: {info['max_capacity']}")
print(f"Utilização: {info['capacity_utilization']:.1%}")
```

5. Integração com frota:
```python
# Usar dados de integração com múltiplos veículos
vehicle_data = {
    'vehicle_capacity': 2,
    'max_vehicles': 3,
    'depot': (0, 0),
    'vehicles_used': 2,
    'unserved_patients': 0
}

is_valid = capacity_restriction.validate_route(route, vehicle_data)
penalty = capacity_restriction.calculate_penalty(route, vehicle_data)
```

## Exemplo Completo

```python
from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction

# Criar restrição de capacidade
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=2)

# Rota com 5 pacientes
route = [
    (0, 0),     # Hospital base
    (10, 10),   # Paciente 1
    (20, 20),   # Paciente 2
    (30, 30),   # Paciente 3
    (40, 40),   # Paciente 4
    (50, 50),   # Paciente 5
    (0, 0)      # Retorno ao hospital
]

# Validação simples (veículo único)
if capacity.validate_route(route):
    print("✅ Rota válida para um veículo")
else:
    print("❌ Rota excede capacidade de um veículo")
    penalty = capacity.calculate_penalty(route)
    print(f"Penalidade: {penalty:.1f}")

# Obter informações de capacidade
info = capacity.get_capacity_info(route)
print(f"\n📊 Informações de capacidade:")
print(f"   Pacientes: {info['patient_count']}")
print(f"   Capacidade máxima: {info['max_capacity']}")
print(f"   Capacidade restante: {info['remaining_capacity']}")
print(f"   Utilização: {info['capacity_utilization']:.1%}")

# Integração com frota
vehicle_data = {
    'vehicle_capacity': 2,
    'max_vehicles': 3,
    'depot': (0, 0),
    'vehicles_used': 3,        # 3 veículos necessários
    'unserved_patients': 0      # todos atendidos
}

print(f"\n🚑 Validação com frota:")
if capacity.validate_route(route, vehicle_data):
    print("✅ Frota pode atender todos os pacientes")
else:
    print("❌ Frota insuficiente para atender todos os pacientes")
```

## Parâmetros Configuráveis

- `max_patients_per_vehicle` (padrão: 10): Capacidade máxima de pacientes por veículo
  - Limite individual de cada ambulância
  - Usado quando não há dados de integração com frota
  - Deve refletir a capacidade real das ambulâncias

## Dados de Integração (`vehicle_data`)

A restrição aceita dados de integração para trabalhar com frota:

### Campos Esperados
- `vehicle_capacity` (int): Capacidade por veículo (override do parâmetro padrão)
- `max_vehicles` (int): Número máximo de veículos disponíveis
- `depot` (tuple): Coordenadas do depósito (excluído da contagem de pacientes)
- `vehicles_used` (int, opcional): Número efetivo de veículos usados
- `unserved_patients` (int, opcional): Pacientes não atendidos

### Comportamento com Integração
- **Com `unserved_patients`**: Prioriza este sinal para validação
- **Com `vehicles_used`**: Valida contra limite de `max_vehicles`
- **Sem sinais específicos**: Calcula distribuição mínima necessária
- **Sem `vehicle_data`**: Modo conservador (não bloqueia rotas)

## Métricas Disponíveis

O método `get_capacity_info()` retorna:
- `patient_count`: Número de pacientes na rota
- `max_capacity`: Capacidade máxima por veículo
- `remaining_capacity`: Capacidade não utilizada
- `capacity_utilization`: Percentual de utilização (0-1)
- `is_within_capacity`: Se a rota respeita a capacidade

## Métodos Principais

### Validação
- `validate_route(route, vehicle_data=None)`: Verifica se a rota respeita a capacidade
- `_validate_multiple_vehicles_distribution()`: Valida distribuição em frota

### Cálculo
- `calculate_penalty(route, vehicle_data=None)`: Calcula penalidade por excesso
- `_get_max_capacity(vehicle_data=None)`: Obtém capacidade efetiva

### Utilitários
- `_extract_patients(route, vehicle_data=None)`: Extrai pacientes (exclui depósito)
- `_has_multiple_vehicles_integration(vehicle_data=None)`: Verifica integração com frota

## Integração com Outras Restrições

Esta restrição é projetada para trabalhar em conjunto com:
- **MultipleVehiclesRestriction**: Recebe dados de frota via `vehicle_data`
- **FuelRestriction**: Pode usar `vehicles_used` para calcular consumo total
- **RouteCostRestriction**: Pode multiplicar custos pelo número de veículos

### Exemplo de Integração Completa
```python
from restrictions.multiple_vehicles import MultipleVehiclesRestriction
from restrictions.vehicle_capacity_restriction import VehicleCapacityRestriction

# Criar restrições
fleet = MultipleVehiclesRestriction(max_vehicles=3, depot=(0,0), vehicle_capacity=2)
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=2)

# Rota de exemplo
route = [(0,0), (10,10), (20,20), (30,30), (40,40), (0,0)]

# Obter dados de integração da frota
vehicle_data = fleet.get_vehicle_data_for_capacity_restriction(route)

# Usar dados na validação de capacidade
is_valid = capacity.validate_route(route, vehicle_data)
penalty = capacity.calculate_penalty(route, vehicle_data)

print(f"Validação com frota: {is_valid}")
print(f"Penalidade: {penalty:.1f}")
print(f"Veículos usados: {vehicle_data.get('vehicles_used', 'N/A')}")
print(f"Pacientes não atendidos: {vehicle_data.get('unserved_patients', 'N/A')}")
```

## Limitações

1. **Modo conservador sem integração**
   - Sem `vehicle_data`, não bloqueia rotas
   - Pode permitir soluções inviáveis em cenários de frota
   - Recomenda-se sempre usar integração quando disponível

2. **Capacidade uniforme**
   - Assume capacidade igual para todos os veículos
   - Para frota heterogênea, seria necessário estender a classe
   - Não considera diferentes tipos de ambulâncias

3. **Sem consideração de peso/volume**
   - Conta apenas número de pacientes
   - Não considera peso, equipamentos ou espaço físico
   - Para restrições físicas, seria necessário nova implementação

4. **Distribuição mínima**
   - Usa apenas cálculo ceil() para distribuição em frota
   - Não considera otimização de distribuição
   - Para distribuição ótima, seria necessário algoritmo mais complexo

## Características Importantes

### Flexibilidade de Integração
- **Modo standalone**: Funciona sem dados de frota
- **Modo integrado**: Usa dados de outras restrições
- **Modo conservador**: Não bloqueia sem contexto suficiente

### Validação Inteligente
- **Prioriza sinais diretos**: `unserved_patients` tem prioridade
- **Fallback para cálculo**: Calcula distribuição quando necessário
- **Considera depósito**: Exclui hospital da contagem de pacientes

### Penalização Proporcional
- **Alta penalidade**: 300 pontos por paciente não atendido
- **Penalidade moderada**: 100 pontos por excesso de capacidade
- **Penalidade de saturação**: Adicional para utilização > 90%

## Dicas de Uso

1. **Configuração básica**:
   - Defina `max_patients_per_vehicle` baseado na capacidade real
   - Use valores conservadores para evitar problemas operacionais
   - Considere margem de segurança para situações especiais

2. **Integração com frota**:
   - Sempre use `vehicle_data` quando disponível
   - Priorize sinais de `unserved_patients` para validação
   - Monitore `vehicles_used` contra `max_vehicles`

3. **Interpretação de métricas**:
   - `capacity_utilization` próximo de 1.0 indica boa eficiência
   - `remaining_capacity` negativo indica excesso
   - `is_within_capacity` False indica violação da restrição

4. **Otimização**:
   - Aumente `vehicle_capacity` para reduzir número de veículos
   - Monitore penalidades para identificar gargalos
   - Use integração para validação mais precisa

## Cenários de Uso

### Cenário 1: Veículo Único
```python
# Para problemas simples com uma ambulância
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=3)
is_valid = capacity.validate_route(route)  # Sem vehicle_data
```

### Cenário 2: Frota com Integração
```python
# Para problemas complexos com múltiplas ambulâncias
fleet = MultipleVehiclesRestriction(max_vehicles=5, depot=(0,0), vehicle_capacity=2)
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=2)

vehicle_data = fleet.get_vehicle_data_for_capacity_restriction(route)
is_valid = capacity.validate_route(route, vehicle_data)
```

### Cenário 3: Validação Conservadora
```python
# Quando não há contexto de frota disponível
capacity = VehicleCapacityRestriction(max_patients_per_vehicle=1)
# Sempre retorna True para validate_route() sem vehicle_data
# Mas ainda calcula penalidades apropriadas
```
