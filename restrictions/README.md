# Restrições do Projeto - Caixeiro Viajante para Rotas Médicas

Esta pasta contém a implementação das restrições adicionais específicas para o problema do caixeiro viajante adaptado para rotas médicas. Aqui são definidas as regras e limitações que tornam nossa solução mais realista e aplicável ao contexto médico.

## Propósito

As restrições implementadas nesta pasta têm como objetivo:

1. Garantir que as rotas geradas sejam viáveis para o contexto médico
2. Considerar fatores específicos como:
   - [ ] Horários de atendimento dos pacientes
   - [ ] Prioridades de atendimento
   - [ ] Tempo máximo de deslocamento
   - [ ] Capacidade de atendimento por período
   - [ ] Restrições de equipamentos médicos
   - [ ] Janelas de tempo para procedimentos específicos

    Legenda:
    * [ ] - Não iniciado;
    * &#10060; - Cancelado;
    * &#10004; - Concluído;
    * &#9989; - Validado;
## Estrutura

Cada restrição deve ser implementada em um arquivo separado, com sua própria documentação explicando:
- O propósito da restrição
- Como ela é aplicada
- Quais parâmetros são considerados
- Como ela afeta o resultado final da rota

## Integração

As restrições aqui definidas serão utilizadas pelo algoritmo principal para gerar rotas que não apenas otimizem a distância total percorrida, mas também atendam a todas as necessidades específicas do contexto médico.
