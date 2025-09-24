import json
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta

class LLMIntegration:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.conversation_history = []
        
    def generate_delivery_instructions(self, route: List[Tuple[float, float]], 
                                     route_info: Dict[str, Any] = None) -> str:
        route_data = {
            "total_cities": len(route),
            "route_coordinates": route,
            "estimated_distance": route_info.get("distance", 0) if route_info else 0,
            "restrictions_summary": route_info.get("restrictions", {}) if route_info else {}
        }
        
        if self.llm_client:
            prompt = self._create_instructions_prompt(route_data, route)
            response = self._call_llm(prompt)
            return response
        else:
            return self._generate_fallback_instructions(route_data)
    
    def generate_route_report(self, routes_data: List[Dict[str, Any]], 
                            period: str = "daily") -> str:
        summary_stats = self._calculate_summary_statistics(routes_data)
        
        if self.llm_client:
            prompt = self._create_report_prompt(summary_stats, routes_data, period)
            response = self._call_llm(prompt)
            return response
        else:
            return self._generate_fallback_report(summary_stats, period)
    
    def answer_route_question(self, question: str, 
                            route_context: Dict[str, Any] = None) -> str:
        if self.llm_client:
            prompt = self._create_question_prompt(question, route_context)
            response = self._call_llm(prompt)
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response,
                "context": route_context
            })
            return response
        else:
            return self._generate_fallback_answer(question, route_context)
    
    def _call_llm(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'chat'):
                response = self.llm_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1500,
                    temperature=0.7
                )
                return response.choices[0].message.content
            else:
                return "LLM client não configurado corretamente."
        except Exception as e:
            return f"Erro ao chamar LLM: {str(e)}"
    
    def _create_instructions_prompt(self, route_data: Dict[str, Any], route: List[Tuple[float, float]]) -> str:
        return f"""
Você é um assistente especializado em logística médica. Gere instruções detalhadas para entrega de medicamentos.

DADOS DA ROTA:
- Total de paradas: {route_data['total_cities']}
- Distância estimada: {route_data['estimated_distance']:.2f} km
- Restrições ativas: {list(route_data['restrictions_summary'].keys())}

COORDENADAS: {self._format_coordinates_for_prompt(route)}

Gere instruções claras e objetivas para a equipe.
"""
    
    def _create_report_prompt(self, stats: Dict[str, float], routes_data: List[Dict[str, Any]], period: str) -> str:
        return f"""
Gere um relatório {period} de logística médica:

ESTATÍSTICAS:
- Rotas executadas: {stats['total_routes']}
- Distância total: {stats['total_distance']:.2f} km
- Eficiência média: {stats['avg_efficiency']:.1f}%

Formate como relatório executivo profissional.
"""
    
    def _create_question_prompt(self, question: str, context: Dict[str, Any]) -> str:
        context_info = ""
        if context:
            context_info = f"CONTEXTO: {context}"
        
        return f"""
{context_info}
PERGUNTA: {question}
Responda de forma técnica e prática sobre logística médica.
"""
    
    def _format_coordinates_for_prompt(self, route: List[Tuple[float, float]]) -> str:
        formatted = []
        for i, (x, y) in enumerate(route, 1):
            formatted.append(f"Parada {i}: ({x:.1f}, {y:.1f})")
        return "\n".join(formatted)
    
    def _calculate_summary_statistics(self, routes_data: List[Dict[str, Any]]) -> Dict[str, float]:
        if not routes_data:
            return {
                'total_routes': 0,
                'total_distance': 0.0,
                'total_time': 0.0,
                'violation_rate': 0.0,
                'avg_efficiency': 0.0
            }
        
        total_routes = len(routes_data)
        total_distance = sum(route.get('distance', 0) for route in routes_data)
        total_time = sum(route.get('time', 0) for route in routes_data)
        violations = sum(1 for route in routes_data if route.get('violations', []))
        avg_efficiency = sum(route.get('efficiency', 0) for route in routes_data) / total_routes
        
        return {
            'total_routes': total_routes,
            'total_distance': total_distance,
            'total_time': total_time,
            'violation_rate': (violations / total_routes) * 100 if total_routes > 0 else 0,
            'avg_efficiency': avg_efficiency
        }
    
    def _generate_fallback_instructions(self, route_data: Dict[str, Any]) -> str:
        return f"""
INSTRUÇÕES DE ENTREGA - ROTA OTIMIZADA

Total de Paradas: {route_data['total_cities']}
Distância Estimada: {route_data['estimated_distance']:.2f} km

SEQUÊNCIA DE ENTREGA:
Siga a ordem exata das coordenadas fornecidas para otimização máxima.

RECOMENDAÇÕES GERAIS:
1. Verificar medicamentos antes da saída
2. Confirmar endereços de entrega
3. Manter medicamentos refrigerados quando necessário
4. Registrar horário de cada entrega
5. Comunicar imediatamente qualquer problema

PROCEDIMENTOS DE EMERGÊNCIA:
- Em caso de urgência médica: contatar central imediatamente
- Para problemas de trânsito: reavaliar rota alternativa
- Medicamentos danificados: não entregar e reportar

OBSERVAÇÃO: Sistema em modo fallback. Configure LLM para instruções detalhadas.
"""
    
    def _generate_fallback_report(self, stats: Dict[str, float], period: str) -> str:
        return f"""
RELATÓRIO {period.upper()} DE ROTAS MÉDICAS

RESUMO EXECUTIVO:
- Total de rotas executadas: {stats['total_routes']}
- Distância total percorrida: {stats['total_distance']:.2f} km
- Eficiência média: {stats['avg_efficiency']:.1f}%
- Taxa de problemas: {stats['violation_rate']:.1f}%

ANÁLISE DE PERFORMANCE:
O sistema de otimização está funcionando adequadamente. 
Recomenda-se monitoramento contínuo para identificar oportunidades de melhoria.

PRÓXIMOS PASSOS:
1. Configurar integração completa com LLM
2. Implementar alertas automáticos
3. Expandir coleta de dados de performance

OBSERVAÇÃO: Relatório em modo fallback. Configure LLM para análises detalhadas.
"""
    
    def _generate_fallback_answer(self, question: str, context: Dict[str, Any]) -> str:
        return f"""
Pergunta recebida: {question}

Sistema de resposta automática ativo.
Para respostas detalhadas e personalizadas, configure a integração com LLM.

Contexto disponível: {len(context) if context else 0} itens de dados.
"""
