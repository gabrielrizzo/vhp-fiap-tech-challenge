# Restrição de Cidade Inicial Fixa

Esta restrição garante que todas as rotas geradas comecem em uma cidade específica (por exemplo, a base hospitalar).

## Propósito

Em um contexto de rotas médicas, é comum que todas as ambulâncias precisem começar seus trajetos a partir de um ponto fixo, como:
- Base hospitalar
- Central de ambulâncias
- Hospital principal

Esta restrição garante que todas as rotas geradas respeitem este requisito operacional.

## Como é Aplicada

A restrição funciona da seguinte forma:

1. Uma cidade inicial é definida através do método `set_start_city()`
2. Toda rota proposta é validada para garantir que começa nesta cidade
3. Rotas que não começam na cidade definida são:
   - Consideradas inválidas pelo método `is_valid()`
   - Recebem penalidade máxima no cálculo de fitness

## Parâmetros

- `start_city`: Tuple[float, float] - Coordenadas (x, y) da cidade inicial obrigatória

## Impacto no Resultado

Esta restrição:
1. Força todas as rotas a começarem no ponto definido
2. Elimina soluções que começam em outros pontos através de penalidades
3. Pode aumentar a distância total da rota em alguns casos, mas garante a viabilidade operacional

## Exemplo de Uso

```python
from restrictions.fixed_start_city import FixedStartCity

# Criar a restrição
fixed_start = FixedStartCity()

# Definir a base hospitalar como ponto inicial (coordenadas x=0, y=0)
fixed_start.set_start_city((0, 0))

# Adicionar ao gerenciador de restrições
multi_restriction.add_restriction("fixed_start", fixed_start)
multi_restriction.activate_restriction("fixed_start")
```
