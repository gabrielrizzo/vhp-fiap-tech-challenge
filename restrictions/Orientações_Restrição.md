# Orientações para Implementar a Restrição: Distância Máxima Total (250 km por Ambulância)

## Propósito da Restrição
Esta restrição garante que nenhuma ambulância percorra mais de 250 km em um único plantão, forçando a divisão dos atendimentos entre múltiplas ambulâncias quando necessário. Após atingir o limite, outra ambulância assume os próximos atendimentos.

## Como Implementar

### 1. Entendimento do Contexto Atual
- O código atual implementa um TSP (Traveling Salesman Problem) clássico com uma única rota.
- Para suportar múltiplas ambulâncias, precisamos adaptar para mTSP (Multiple Traveling Salesmen Problem).
- A restrição assume que todas as rotas começam e terminam na base hospitalar (cidade inicial fixa).

### 2. Alterações Necessárias

#### a) Modificar a Estrutura de Dados das Soluções
- **Arquivo a alterar:** `genetic_algorithm.py`
- **O que alterar:** Mudar as soluções de `List[List[Tuple[float, float]]]` (uma rota) para `List[List[List[Tuple[float, float]]]]` (lista de rotas, uma para cada ambulância).
- Cada rota interna deve começar e terminar na base (primeira cidade).

#### b) Atualizar Função de Fitness
- **Arquivo a alterar:** `genetic_algorithm.py` (função `calculate_fitness`)
- **O que alterar:** Modificar para calcular a soma das distâncias de todas as rotas das ambulâncias.
- Adicionar penalização se qualquer rota exceder 250 km.

Exemplo de código:
```python
def calculate_fitness_with_constraints(routes: List[List[Tuple[float, float]]], max_distance: float = 250.0) -> float:
    total_distance = 0
    penalty = 0
    base = routes[0][0]  # Assume base is the first city
    
    for route in routes:
        route_distance = calculate_route_distance(route)
        total_distance += route_distance
        if route_distance > max_distance:
            penalty += (route_distance - max_distance) * 1000  # Penalização alta
    
    return total_distance + penalty

def calculate_route_distance(route: List[Tuple[float, float]]) -> float:
    distance = 0
    for i in range(len(route) - 1):
        distance += calculate_distance(route[i], route[i+1])
    return distance
```

#### c) Atualizar Operadores Genéticos
- **Crossover:** Adaptar `order_crossover` para trabalhar com múltiplas rotas.
- **Mutation:** Modificar `mutate` e `mutate_hard` para operar em cada rota individualmente.
- **Seleção:** Garantir que a seleção considere as novas estruturas.

#### d) Inicialização da População
- **Arquivo a alterar:** `genetic_algorithm.py` (função `generate_random_population`)
- **O que alterar:** Gerar populações com múltiplas rotas. Por exemplo, dividir as cidades aleatoriamente entre ambulâncias, garantindo que cada rota não exceda 250 km inicialmente.

#### e) Visualização
- **Arquivo a alterar:** `tsp.py` ou `tsp_diversity_control.py`
- **O que alterar:** Modificar `draw_paths` para desenhar múltiplas rotas com cores diferentes para cada ambulância.

### 3. Parâmetros da Restrição
- **Distância máxima por ambulância:** 250 km
- **Base:** Cidade inicial (índice 0)
- **Penalização:** Aplicar penalização alta (ex: multiplicar excesso por 1000) para rotas que excedem o limite.

### 4. Integração com Outras Restrições
- Esta restrição deve ser combinada com outras como "Cidade inicial fixa" e "mTSP".
- Certifique-se de que as rotas começam na base e não violam outras restrições.

### 5. Testes
- Testar com poucos cidades para validar que as rotas são divididas corretamente.
- Verificar que a penalização força a divisão quando necessário.

### 6. Próximos Passos
- Implementar o mTSP básico primeiro (Gerson está responsável).
- Depois integrar esta restrição.
- Documentar no README.md da pasta restrictions/ quando concluído.

## Padrão dos Colegas
- Criar arquivo separado: `max_distance_constraint.py` na pasta `restrictions/`.
- Incluir docstring explicando propósito, parâmetros e impacto.
- Seguir estrutura similar às outras restrições implementadas.