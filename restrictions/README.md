# Restrições do Projeto - Caixeiro Viajante para Rotas Médicas

Esta pasta contém a implementação das restrições adicionais específicas para o problema do caixeiro viajante adaptado para rotas médicas. Aqui são definidas as regras e limitações que tornam nossa solução mais realista e aplicável ao contexto médico.

## Propósito

As restrições implementadas nesta pasta têm como objetivo:

1. Garantir que as rotas geradas sejam viáveis para o contexto médico
2. Considerar fatores específicos como:
   - ⬜ Distância máxima total – Uma ambulância não pode rodar mais que 250 km em um plantão, pois precisa retornar à base para manutenção preventiva. Exemplo: após atingir o limite, outra ambulância deve assumir os próximos atendimentos.
   - ⬜ Tempo máximo total – A equipe deve concluir sua rota em até 8 horas de plantão. Exemplo: após esse tempo, a ambulância precisa voltar para troca de equipe médica.
   - ✅ Rotas proibidas – Certas ruas estão interditadas por enchentes ou obras. Exemplo: a ambulância não pode usar a Avenida Central, pois está alagada. (Vinicius)
   - ✅ Rotas unidirecionais – Algumas vias são de mão única. Exemplo: a Rua da Saúde só pode ser usada no sentido bairro → centro, forçando a ambulância a dar uma volta maior. (Vinicius)
   - ⬜ Orçamento máximo – Há um limite de combustível e insumos para cada ambulância em uma operação de emergência. Exemplo: a rota não pode gastar mais que R$ 500 em combustível em um turno.
   - 🔄 Custos diferenciados por rota – Algumas estradas têm pedágio urbano ou grande congestionamento.  Exemplo: o sistema calcula que passar pelo túnel custará mais (tempo + pedágio), podendo evitar essa opção. (Gabriel)
   - ⬜ Cidade inicial fixa – A ambulância sempre deve sair da base hospitalar.  Exemplo: toda rota começa obrigatoriamente no Hospital Municipal.
   - ⬜ Cidades prioritárias – Pacientes em estado crítico precisam ser atendidos antes dos demais.  Exemplo: parada cardíaca antes de transporte de paciente para exame.
   - ⬜ Checkpoints obrigatórios – A ambulância deve obrigatoriamente passar por uma farmácia hospitalar para coletar medicamentos antes de seguir para outro paciente.
   - ⬜ Distância máxima entre paradas – A cada 50 km a ambulância precisa parar em algum ponto para checagem rápida (água, reabastecimento parcial, troca de oxigênio).
   - ⬜ Tempo de atendimento fixo em cada cidade – Cada visita leva pelo menos 20 minutos de estabilização do paciente antes de seguir viagem.
   - ⬜ Limite de visitas por cidade – A ambulância só pode passar uma vez em determinado bairro, para evitar sobreposição de atendimentos desnecessários.
   - ⬜ Janelas de tempo (time windows) – Um paciente precisa tomar medicação intravenosa entre 14h e 15h. Exemplo: a ambulância deve programar a rota para chegar exatamente nesse intervalo.
   - ⬜ Capacidade máxima do veículo – A ambulância só pode visitar N cidades no dia.
   - ⬜ Abastecimento obrigatório – Após rodar 180 km, a ambulância precisa obrigatoriamente parar em um posto de combustível credenciado.
   - ⬜ Custos dinâmicos – Em horário de pico, certas vias têm tempo de deslocamento 3x maior.  Exemplo: atravessar o centro às 18h gera alto custo de tempo, logo o sistema prefere rotas alternativas.
   - ⬜ Visitas múltiplas obrigatórias – O mesmo paciente precisa de duas visitas: coleta em casa e transporte ao hospital.
   - ⬜ Capacidade de atendimento por período
   - 🔄 Restrição Muito Avançada Múltiplos caixeiros (mTSP) – Várias ambulâncias dividem os atendimentos entre diferentes pacientes espalhados pela cidade. (Gerson)
   - ⬜ Balanceamento de rotas – Nenhuma ambulância deve ficar sobrecarregada com 10 atendimentos enquanto outra faz apenas 2.
   - ⬜ Áreas exclusivas por caixeiro – Uma ambulância só pode operar na Zona Sul, enquanto outra cobre a Zona Norte. Exemplo: reduzir tempo de resposta criando zonas de responsabilidade.


### Legenda: 
   * ⬜ Não iniciado
   * 🔄 Em andamento
   * ✅ Concluído
   * ❌ Cancelado
   * 🛑 Bloqueado

## Estrutura

Cada restrição deve ser implementada em um arquivo separado, com sua própria documentação explicando:
- O propósito da restrição
- Como ela é aplicada
- Quais parâmetros são considerados
- Como ela afeta o resultado final da rota

## Restrições Implementadas

Aqui está a lista de restrições já implementadas com links para suas documentações detalhadas:

1. [Rotas Proibidas](./forbidden_routes/README.md) - Implementa restrições para rotas interditadas por enchentes, obras, etc.
2. [Rotas Unidirecionais](./one_way_routes/README.md) - Implementa restrições para vias de mão única, onde só é possível trafegar em uma direção específica

## Integração

As restrições aqui definidas serão utilizadas pelo algoritmo principal para gerar rotas que não apenas otimizem a distância total percorrida, mas também atendam a todas as necessidades específicas do contexto médico.
