# 🚑 Medical Route TSP Optimizer - FIAP Tech Challenge Fase 2 Grupo 

Sistema avançado de otimização de rotas médicas usando Algoritmos Genéticos com restrições realistas e integração com LLM para geração de relatórios inteligentes.

## 👥 Equipe - FIAP Tech Challenge

- **Bruno**: RM
- **Gerson**: RM
- **Gabriel**: RM
- **Amorin**: RM
- **Mauricio**: RM
  
## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Uso](#-uso)
- [Restrições Implementadas](#-restrições-implementadas)
- [Datasets](#-datasets)
- [Configuração](#-configuração)
- [Testes](#-testes)
- [Equipe](#-equipe)

## 🎯 Visão Geral

Este projeto implementa um sistema completo de otimização de rotas para veículos médicos (ambulâncias) usando Algoritmos Genéticos avançados. O sistema considera múltiplas restrições do mundo real e oferece visualização em tempo real do processo de otimização.

### ✅ Status do Projeto - COMPLETO

- **Algoritmo Genético**: Sistema completo com operadores especializados
- **7 Restrições Médicas**: Sistema modular para constraints realistas  
- **Integração LLM**: Sistema completo de IA para instruções e relatórios
- **Visualização**: Interfaces Pygame e Streamlit com mapas reais
- **Documentação**: README detalhado + documentação inline
- **Testes**: Suite automatizada de validação

## ✨ Funcionalidades

### Core do Algoritmo Genético
- **População adaptativa**: 2000 indivíduos com diversidade controlada
- **Operadores genéticos**: Crossover ordenado, mutação adaptativa
- **Seleção inteligente**: Torneio e rank-based com awareness de diversidade
- **Fases de execução**: Exploração inicial + Refinamento posterior
- **Elitismo**: Preservação das melhores soluções

### Visualização e Interface
- **Interface Pygame**: Visualização em tempo real da evolução
- **Interface Streamlit**: Dashboard web interativo com mapas
- **Suporte a mapas reais**: Integração com dados de hospitais de São Paulo
- **Gráficos de evolução**: Acompanhamento do fitness ao longo das gerações

## 🏗 Arquitetura

```
medical-tsp-fiap/
├── core/                           # Núcleo do algoritmo genético
│   ├── enhanced_genetic_algorithm.py
│   ├── restriction_manager.py
│   ├── config_manager.py
│   └── base_restriction.py
├── restrictions/                   # Restrições implementadas
│   ├── fuel_restriction.py        # Limite de combustível
│   ├── vehicle_capacity_restriction.py  # Capacidade de pacientes
│   ├── route_cost_restriction.py  # Custos de rota (pedágios)
│   ├── multiple_vehicles.py       # Múltiplas ambulâncias
│   ├── fixed_start_restriction.py # Início fixo (hospital)
│   ├── forbidden_routes.py        # Rotas proibidas
│   └── one_way_routes.py          # Rotas de mão única
├── llm/                           # Integração com IA
│   └── llm_integration.py        # GPT para relatórios
├── utils/                         # Utilitários
│   ├── draw_functions.py         # Funções de desenho
│   ├── helper_functions.py       # Funções auxiliares
│   ├── route_utils.py            # Utilitários de rota otimizados
│   └── selection_functions.py    # Funções de seleção
├── data/                          # Datasets
│   ├── benchmark_att48.py        # Benchmark ATT48
│   └── benchmark_hospitals_sp.py # Hospitais de SP
├── config/                        # Configurações
│   ├── medical_tsp_config.json   # Config principal
│   └── route_cost.py             # Custos específicos
├── tests/                         # Testes automatizados
│   ├── test_medical_tsp.py
│   ├── test_ambulance_patient_tsp.py
│   ├── test_capacity_integration.py
│   ├── test_forbidden_routes.py
│   ├── test_one_way_routes.py
│   └── test_multiple_vehicles_integration.py
├── docs/
│   └── restrictions/              # Documentação das restrições
│       ├── fuel_restriction.md
│       ├── forbidden_routes.md
│       ├── one_way_routes.md
│       └── route_cost_restriction.md
├── main_tsp_medical.py           # App principal (Pygame)
├── medical_tsp_streamlit.py     # App web (Streamlit)
├── requirements.txt              # Dependências
└── README.md                     
```

## 🚀 Instalação

### Pré-requisitos
- Python 3.8+
- pip

### Setup do Ambiente

```bash
# Clone o repositório
git clone https://github.com/gabrielrizzo/vhp-fiap-tech-challenge.git
cd vhp-fiap-tech-challenge

# Crie um ambiente virtual
python -m venv venv

# Ative o ambiente virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale as dependências
pip install -r requirements.txt

# Configure variáveis de ambiente (opcional para LLM)
cp .env.example .env
# Edite .env e adicione sua OPENAI_API_KEY
```

## 💻 Uso

### Interface Principal (Pygame)

```bash
python main_tsp_medical.py
```

**Controles:**
- `Q` - Sair
- `R` - Gerar relatório de rota
- `I` - Mostrar instruções da rota
- `C` - Mostrar configuração atual

### Interface Web (Streamlit) - RECOMENDADO

```bash
streamlit run medical_tsp_streamlit.py
```

Acesse: http://localhost:8501

**Funcionalidades:**
- Escolha entre datasets (ATT48 ou Hospitais SP)
- Configure restrições dinamicamente via sidebar
- Visualize evolução em tempo real
- Execute geração por geração ou todas de uma vez
- Gere relatórios automatizados

## 🔧 Restrições Implementadas

### 1. ⛽ Restrição de Combustível
Limita a distância máxima e calcula custos de combustível.

📖 [**Documentação Detalhada**](docs/restrictions/fuel_restriction.md)

**Configuração:**
- `max_distance`: Distância máxima em km (padrão: 250)
- `fuel_cost_per_km`: Custo por km (padrão: 0.8)
- `fuel_cost_limit`: Limite de custo total

---

### 2. 🚗 Capacidade do Veículo
Limita o número de pacientes por veículo.

**Configuração:**
- `max_patients_per_vehicle`: Pacientes por veículo (padrão: 10)
- Integração automática com múltiplos veículos

---

### 3. 💰 Custo de Rotas (Pedágios)
Adiciona custos específicos para certas rotas.

📖 [**Documentação Detalhada**](docs/restrictions/route_cost_restriction.md)

**Configuração:**
- `route_cost_dict`: Dicionário com custos por rota
- Simula pedágios, taxas de travessia, etc.

---

### 4. 🚑 Múltiplos Veículos
Distribui pacientes entre várias ambulâncias.

**Configuração:**
- `max_vehicles`: Número máximo de veículos (padrão: 5)
- `vehicle_capacity`: Capacidade por veículo
- `depot`: Local do hospital base

---

### 5. 🏥 Início Fixo (Hospital)
Força as rotas a começarem no hospital.

**Configuração:**
- `hospital_location`: Coordenadas do hospital
- Penalidade alta para rotas inválidas (5000)

---

### 6. 🚫 Rotas Proibidas
Define rotas que não podem ser utilizadas.

📖 [**Documentação Detalhada**](docs/restrictions/forbidden_routes.md)

**Configuração:**
- `base_distance_penalty`: Penalidade base (padrão: 1000)
- Simula ruas interditadas, enchentes, obras

---

### 7. ➡️ Rotas Unidirecionais (Mão Única)
Define rotas que só podem ser percorridas em uma direção.

📖 [**Documentação Detalhada**](docs/restrictions/one_way_routes.md)

**Configuração:**
- `base_distance_penalty`: Penalidade para contramão (padrão: 2000)
- Simula ruas de mão única

## 📊 Datasets

### ATT48 Benchmark
- 48 cidades com solução ótima conhecida
- Coordenadas em pixels
- Usado para validação do algoritmo

### Hospitais de São Paulo
- 48 hospitais reais de São Paulo
- Coordenadas geográficas (lat/lon)
- Visualização com mapa real via Mapbox
- Alguns hospitais que estavam sem nome, utilizamos nomes fictícios para exibição (Albert Einstein)

## ⚙️ Configuração

### Arquivo de Configuração
O sistema usa `config/medical_tsp_config.json` para todas as configurações:

```json
{
    "genetic_algorithm": {
        "population_size": 2000,
        "generation_limit": 10000,
        "n_cities": 48
    },
    "mutation": {
        "initial_probability": 0.85,
        "initial_intensity": 25
    },
    "restrictions": {
        "fuel": {
            "enabled": true,
            "max_distance": 250.0,
            "fuel_cost_per_km": 0.8
        },
        // ... outras restrições
    },
    "llm": {
        "enabled": true,
        "model": "gpt-3.5-turbo",
        "fallback_mode": true
    }
}
```

### Configuração Dinâmica (Streamlit)
Na interface Streamlit, você pode configurar restrições dinamicamente via sidebar sem editar arquivos.

## 🧪 Testes

### Executar Todos os Testes
```bash
# Teste principal
python tests/test_medical_tsp.py

# Testes específicos
python tests/test_ambulance_patient_tsp.py
python tests/test_capacity_integration.py
python tests/test_forbidden_routes.py
python tests/test_one_way_routes.py
python tests/test_multiple_vehicles_integration.py
```

### Cobertura de Testes
- ✅ Importações e estrutura básica
- ✅ Funcionalidade das restrições individuais
- ✅ Integração entre restrições
- ✅ Cálculo de fitness e penalidades
- ✅ Validação de rotas

## 🤖 Integração LLM

### Configuração
1. Obtenha uma API key da OpenAI
2. Configure no arquivo `.env`:
```env
OPENAI_API_KEY=sua_chave_aqui
```

### Funcionalidades com LLM
- **Instruções de Entrega**: Orientações detalhadas para motoristas
- **Relatórios de Performance**: Análise automática de métricas
- **Respostas a Perguntas**: Suporte contextual sobre rotas

### Modo Fallback
Se não configurar a API key, o sistema funciona normalmente com respostas pré-definidas.

## 📈 Métricas e Performance

### Otimizações Implementadas
- Cache LRU para cálculo de distâncias
- RouteUtils centralizado para operações comuns
- Diversidade populacional controlada
- Mutação adaptativa baseada em progresso

### Resultados Típicos
- **ATT48**: Convergência em ~1000 gerações
- **Hospitais SP**: Otimização efetiva com restrições reais
- **Performance**: ~30 FPS na visualização Pygame

## 📝 Licença

Projeto acadêmico - FIAP Tech Challenge Fase 2 - IA para Devs

## 📚 Recursos Adicionais

- [Guia de Restrições](docs/restrictions/)

---

**Projeto desenvolvido para o Tech Challenge da FIAP - Fase 2**

*Última atualização: Outubro 2025*