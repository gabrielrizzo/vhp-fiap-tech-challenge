# Explicação das Implementações - Restrição de Distância Máxima

## 1. Arquivo: `max_distance_constraint.py`

### Propósito
Implementa as funções específicas para aplicar a restrição de distância máxima de 250 km por ambulância no contexto de múltiplas rotas (mTSP).

### Principais Funções
- `apply_max_distance_constraint(routes, max_distance, penalty_factor)`: Calcula a penalização total para rotas que excedem o limite.
- `calculate_route_distance(route)`: Calcula a distância de uma única rota.
- `is_route_valid(route, max_distance)`: Verifica se uma rota está dentro do limite.
- `split_route_by_distance(full_route, max_distance)`: Divide uma rota completa em segmentos válidos.

### Como Funciona
- Para cada rota, calcula a distância total percorrida.
- Se a distância exceder 250 km, aplica uma penalização proporcional ao excesso.
- A penalização é adicionada ao fitness total, forçando o algoritmo a evitar soluções inválidas.

### Integração
- Chamado pela função `calculate_fitness_mtsp` no `genetic_algorithm_max_distance.py`.
- Permite que o AG evolua soluções que respeitam a restrição.

## 2. Arquivo: `genetic_algorithm_max_distance.py`

### Propósito
Adaptação do algoritmo genético clássico para resolver o problema do caixeiro viajante com múltiplas ambulâncias (mTSP), incorporando a restrição de distância máxima.

### Principais Mudanças
- **Estrutura de Dados**: Indivíduos agora são `List[List[List[Tuple]]]`, representando múltiplas rotas.
- **Geração Inicial**: `generate_random_population_mtsp` divide cidades aleatoriamente entre ambulâncias, garantindo que cada rota comece e termine na base.
- **Fitness**: `calculate_fitness_mtsp` soma distâncias de todas as rotas + penalizações por violação de restrição.
- **Crossover**: `order_crossover_mtsp` aplica crossover em cada rota correspondente entre pais.
- **Mutação**: `mutate_mtsp` aplica mutação em cada rota individualmente.

### Constantes Definidas
- `N_AMBULANCES = 3`: Número de ambulâncias.
- `MAX_DISTANCE_PER_AMBULANCE = 250.0`: Limite de distância por ambulância.

### Como Funciona
- Cada ambulância tem sua própria rota, começando e terminando na base hospitalar.
- O fitness penaliza soluções onde qualquer rota exceda 250 km.
- Operadores genéticos preservam a estrutura de múltiplas rotas.

## 3. Arquivo: `tsp_maxdistance.py`

### Propósito
Interface visual e de execução do algoritmo genético adaptado para mTSP com restrição de distância máxima, baseado no `tsp_diversity_control.py`.

### Principais Mudanças
- **População**: Usa funções do `genetic_algorithm_max_distance.py` para gerar e manipular indivíduos com múltiplas rotas.
- **Visualização**: `draw_routes` desenha cada rota com uma cor diferente para distinguir ambulâncias.
- **Diversidade**: Adaptada para considerar a diversidade das rotas (usa a primeira rota como proxy).
- **Injeção de Diversidade**: Funções `inject_random_individuals_mtsp` e `inject_heuristic_individuals_mtsp` para manter diversidade na população mTSP.

### Cores das Rotas
- Azul, Verde, Amarelo, Roxo, Laranja para distinguir as rotas das ambulâncias.

### Como Funciona
- Executa o AG por gerações, mostrando em tempo real as melhores rotas encontradas.
- Penaliza soluções inválidas, evoluindo para rotas que respeitam o limite de 250 km por ambulância.
- Permite visualizar como as rotas são divididas entre ambulâncias.

## 4. Arquivo: `draw_functions.py` (Modificação)

### Mudança Adicionada
- `draw_routes(screen, routes, colors, width)`: Desenha múltiplas rotas com cores diferentes.

### Propósito
Suporte visual para múltiplas rotas no mTSP.

## Integração Geral

1. **Inicialização**: Cidades são divididas entre ambulâncias, cada uma começando na base.
2. **Avaliação**: Fitness = soma das distâncias + penalização por excesso.
3. **Evolução**: Crossover e mutação preservam estrutura de múltiplas rotas.
4. **Restrição**: Penalização força respeito ao limite de 250 km.
5. **Visualização**: Rotas coloridas mostram divisão do trabalho entre ambulâncias.

## Benefícios da Implementação

- **Realismo**: Simula cenário médico com múltiplas ambulâncias e limite de distância.
- **Escalabilidade**: Fácil ajustar número de ambulâncias e limite de distância.
- **Flexibilidade**: Pode ser estendido para outras restrições (tempo, custo, etc.).
- **Visual**: Interface mostra claramente como o problema é resolvido.

## Como Executar

1. Execute `tsp_maxdistance.py` para ver o algoritmo em ação.
2. Ajuste constantes em `genetic_algorithm_max_distance.py` conforme necessário.
3. Monitore o fitness: valores baixos indicam soluções boas e válidas.

## Próximos Passos

- Integrar com outras restrições (tempo, custos, etc.).
- Otimizar operadores genéticos para mTSP.
- Adicionar métricas específicas para avaliação de soluções médicas.