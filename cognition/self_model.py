# prometheus_agi/cognition/self_model.py
from typing import Dict, Any, List


# Se elimina la importación ""

class SelfModel:
    """Modelo interno que la IA tiene de sí misma."""
    # Se cambia la anotación de tipo a un string para romper la dependencia circular
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind
        self.performance_log: Dict[str, List[float]] = {}

    def update_performance_log(self, intent: str, score: float):
        """Añade una nueva puntuación de rendimiento para una intención."""
        self.performance_log.setdefault(intent, []).append(score)
        # Mantener solo los últimos 20 scores para evitar que la lista crezca indefinidamente
        self.performance_log[intent] = self.performance_log[intent][-20:]


class MetacognitionEngine:
    """Analiza el rendimiento y evoluciona los perfiles de fitness."""
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind

    def analyze_and_evolve_profiles(self, batch_results: List[Dict]):
        """Revisa los resultados de un lote y muta los perfiles de fitness."""
        print("\n===== CICLO DE METACOGNICIÓN: EVOLUCIONANDO CRITERIOS DE JUICIO =====")
        results_by_intent = {}
        for res in batch_results:
            intent = res.get('intent')
            if intent and 'score' in res:
                results_by_intent.setdefault(intent, []).append(res['score'])

        for intent, scores in results_by_intent.items():
            if not scores: continue
            avg_score = sum(scores) / len(scores)
            profile = self.mind.get_profile_for_intent(intent)
            self.mind.self_model.update_performance_log(intent, avg_score)
            
            # El rendimiento bajo aumenta la tasa de mutación
            learning_rate = 0.5 * (1 - (avg_score / 1000.0))
            print(f"  Intención '{intent}': Rendimiento promedio={avg_score:.2f}. Tasa de mutación del perfil={learning_rate:.3f}")
            profile.mutate(learning_rate)
        print("====================== METACOGNICIÓN FINALIZADA ======================\n")


class FailureAnalysisEngine:
    """Realiza un análisis post-mortem de los fracasos."""
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind

    def analyze_last_failure(self) -> str:
        """Analiza la última secuencia de fallo y devuelve un diagnóstico."""
        failure_context = self.mind.episodic_memory.get_last_failed_goal_context()
        if not failure_context:
            return "No encuentro registros de fallos recientes para analizar."
        
        failed_goal_details = failure_context[-1]["details"]
        return f"Diagnóstico de fallo para el objetivo '{failed_goal_details.get('description', 'desconocido')}': Causa no determinada (lógica de análisis pendiente)."