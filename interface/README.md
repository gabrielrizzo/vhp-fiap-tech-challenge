# Interface Gráfica - Otimização de Rotas Médicas

Esta pasta contém a implementação da interface gráfica para visualização e controle do algoritmo de otimização de rotas médicas.

## Funcionalidades

1. **Controle de Parâmetros do Algoritmo Genético**
   - Número de cidades
   - Tamanho da população
   - Número de gerações
   - Probabilidade de mutação

2. **Seleção de Métodos**
   - Escolha do método de crossover (Order Crossover ou Uniform Crossover)
   - Escolha do método de mutação (Simple Mutation ou Hard Mutation)
   - Configuração da intensidade da mutação (para Hard Mutation)

3. **Controle de Restrições**
   - Ativação/desativação de restrições
   - Configuração de parâmetros específicos para cada restrição

4. **Visualização em Tempo Real**
   - Mapa interativo mostrando:
     - Localização das cidades
     - Melhor rota encontrada (verde)
     - Rota atual sendo testada (azul tracejado)
     - Rotas proibidas (vermelho)
   - Informações sobre:
     - Geração atual
     - Melhor fitness
     - Sequência da melhor rota

## Como Usar

1. Execute o Streamlit:
   ```bash
   streamlit run interface/app.py
   ```

2. A interface será aberta no seu navegador padrão

3. Configure os parâmetros desejados na barra lateral

4. Clique em "Iniciar Otimização" para começar o processo

5. Observe a evolução do algoritmo em tempo real no mapa e nas informações

## Requisitos

- Python 3.8+
- Streamlit
- Plotly
- NumPy

## Estrutura de Arquivos

- `app.py`: Implementação principal da interface
- `README.md`: Esta documentação
