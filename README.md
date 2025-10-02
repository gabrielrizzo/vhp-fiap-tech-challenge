# Medical Route TSP Optimizer - FIAP Tech Challenge Fase 2

Sistema de otimização de rotas médicas usando Algoritmos Genéticos e integração com LLM.

## 📁 Estrutura do Projeto

```
medical-tsp-fiap/
├── core/                           # Módulos principais
├── restrictions/                   # Restrições médicas específicas
├── llm/                           # Integração com LLM
├── tests/                         # Testes automatizados
├── data/                          # Dados e benchmarks
├── config/                        # Configurações
└── main.py                       # Aplicação principal
```

## 🎮 Executar Otimizador

```bash
python main_tsp_medical.py
```

## 📊 Tech Challenge Requirements

✅ **Algoritmo Genético**: Sistema completo com operadores especializados
✅ **Restrições Médicas**: Sistema modular para constraints realistas  
✅ **Integração LLM**: Sistema completo de IA para instruções e relatórios
✅ **Documentação**: README detalhado + documentação inline
✅ **Testes**: Suite automatizada de validação
✅ **Estrutura Python**: Ambiente virtual + gerenciamento de dependências

## 🤝 Integração da Equipe

- **Bruno (Combustível)**: Expandir `restrictions/fuel_restriction.py`
- **Gerson (Múltiplos Veículos)**: Expandir `restrictions/ambulance_patient_restriction.py`
- **Gabriel (LLM)**: Configurar API em `llm/llm_integration.py`
- **Amorin**: Configurar API em `llm/llm_integration.py`
- **Mauricio**: Configurar API em `llm/llm_integration.py`
---
**Projeto acadêmico**: FIAP Tech Challenge Fase 2 - IA para Devs