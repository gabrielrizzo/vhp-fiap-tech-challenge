# Restrição: Custos de Rotas Médicas (Pedágios e Custos Operacionais)

## Descrição
Esta restrição implementa a funcionalidade de custos de rotas para serviços médicos de emergência, adicionando penalidades financeiras (como pedágios, custos de combustível e taxas operacionais) para rotas específicas entre pontos de atendimento, otimizando tanto a distância total quanto os custos operacionais do percurso.

### Contexto do Problema Médico
- **Custos de pedágio** em rotas específicas entre hospitais e pontos de atendimento
- **Custos operacionais** variáveis por tipo de rota (urbana, rodoviária, rural)
- **Rotas prioritárias** com custos reduzidos para emergências
- **Otimização bi-objetivo**: tempo de resposta + custos operacionais
- **Flexibilidade** para definir custos customizados por rota médica

## Como Funciona
O sistema de otimização é implementado através da classe `RouteCostRestriction` que:
1. **Mapeia custos** de rotas específicas entre pontos médicos usando dicionário de coordenadas
2. **Calcula custos totais** somando pedágios e custos operacionais de cada rota percorrida
3. **Integra com fitness original** adicionando custos ao tempo de deslocamento
4. **Suporta normalização** de coordenadas para compatibilidade com sistemas de geolocalização

### Algoritmos Utilizados
- **Busca em dicionário** para custos de rotas médicas específicas
- **Cálculo de distância euclidiana** para tempo base de deslocamento
- **Normalização de coordenadas** para compatibilidade com mapas digitais
- **Função de fitness bi-objetivo** para balanceamento tempo + custos

## Funcionalidades Principais
### Classe RouteCostRestriction
```python
class RouteCostRestriction:
    def __init__(self, medical_points: List[Tuple[float, float]], route_costs: Dict)
    def calculate_fitness_with_route_cost(path, original_fitness, use_normalized=False) -> float
    def united_fitness_with_route_cost(path, use_normalized=False) -> float
    def get_route_cost(point1, point2) -> float
    def add_route_cost(point1, point2, cost) -> None
    def remove_route_cost(point1, point2) -> None
```

### Funções de Otimização
- `calculate_fitness_with_route_cost()` - Calcula fitness com custos de rotas médicas
- `united_fitness_with_route_cost()` - Função unificada que calcula tempo + custos
- `get_route_cost()` - Obtém custo de uma rota médica específica

## Como Usar
### 1. Configuração Básica para Serviços Médicos
```python
from restrictions.cost_restriction.route_cost_restriction import RouteCostRestriction

# Define custos de pedágio e operacionais para rotas médicas
medical_route_costs = {
    ((hospital_x, hospital_y), (clinic_x, clinic_y)): 15.50,  # Pedágio + combustível
    ((clinic_x, clinic_y), (emergency_x, emergency_y)): 0,    # Rota prioritária gratuita
    ((hospital_x, hospital_y), (rural_x, rural_y)): 45.00,    # Rota rural com custos elevados
}

# Cria restrição de custos para pontos médicos
route_restriction = RouteCostRestriction(medical_points, medical_route_costs)
```

### 2. Configuração para Sistema de Geolocalização
```python
# Configura dimensões para normalização (necessário para mapas digitais)
route_restriction.config_dimensions(
    width=1500, 
    plot_x_offset=450, 
    height=800, 
    node_radius=10
)

# Calcula fitness com coordenadas normalizadas
fitness = route_restriction.united_fitness_with_route_cost(
    path=medical_route,
    use_normalized=True
)
```

### 3. Gestão Dinâmica de Custos Médicos
```python
# Adiciona novo custo de rota médica
route_restriction.add_route_cost((hospital1_x, hospital1_y), (hospital2_x, hospital2_y), 25.75)

# Remove custo de rota (torna gratuita)
route_restriction.remove_route_cost((clinic_x, clinic_y), (pharmacy_x, pharmacy_y))

# Consulta custo de rota médica específica
cost = route_restriction.get_route_cost((hospital_x, hospital_y), (clinic_x, clinic_y))
print(f"Custo da rota médica: R$ {cost:.2f}")
```

## Métricas Retornadas
- **Fitness total**: Tempo de deslocamento + custos operacionais
- **Custo de rotas**: Soma dos pedágios e custos operacionais de todas as rotas percorridas
- **Rotas com custo**: Número de rotas que possuem pedágio/custo definido
- **Rotas prioritárias**: Rotas sem custo definido (emergências médicas)

## Exemplo Completo para Serviços Médicos
Veja o arquivo `demo_cost_restriction.py` para um exemplo completo que demonstra:
- **Integração com algoritmo genético** para otimização de rotas médicas com custos
- **Configuração de pedágios** para rotas específicas entre pontos médicos
- **Cálculo de fitness unificado** considerando tempo + custos operacionais
- **Visualização em tempo real** da evolução do algoritmo de otimização

## Parâmetros Configuráveis
- `medical_points`: Lista de coordenadas dos pontos médicos (hospitais, clínicas, UPA)
- `route_costs`: Dicionário com custos de rotas específicas entre pontos médicos
- `use_normalized`: Se True, usa coordenadas normalizadas (para mapas digitais)
- `width`, `height`, `plot_x_offset`, `node_radius`: Dimensões para normalização

## Formato do Dicionário de Custos Médicos
```python
medical_route_costs = {
    ((hospital1_x, hospital1_y), (hospital2_x, hospital2_y)): 20.00,  # Pedágio entre hospitais
    ((hospital_x, hospital_y), (clinic_x, clinic_y)): 8.50,           # Rota hospital-clínica
    ((clinic_x, clinic_y), (emergency_x, emergency_y)): 0,            # Rota prioritária gratuita
    ((hospital_x, hospital_y), (rural_x, rural_y)): 35.00,            # Rota rural com custos elevados
    # ... mais rotas médicas
}
```

## Casos de Uso Médicos
- **Ambulâncias de emergência**: Otimização considerando pedágios e custos de combustível
- **Serviços de home care**: Planejamento de rotas considerando custos operacionais
- **Distribuição de medicamentos**: Minimização de custos de entrega entre farmácias e hospitais
- **Análise de viabilidade**: Comparação de rotas médicas por custo total vs. tempo de resposta
- **Planejamento de recursos**: Otimização de custos para múltiplas ambulâncias
- **Logística hospitalar**: Gestão de custos para transporte de equipamentos médicos

## Benefícios para Serviços Médicos
- **Redução de custos operacionais** através de otimização de rotas
- **Melhoria na eficiência** do atendimento médico de emergência
- **Planejamento estratégico** de recursos e infraestrutura
- **Análise de viabilidade** para novos pontos de atendimento
- **Otimização multi-objetivo** considerando tempo de resposta e custos