# prometheus_agi/genes/introspective.py
import inspect
import ast
from collections import Counter
from typing import TYPE_CHECKING

# Importaciones del proyecto
from core.base import Gene, ExecutionContext
from knowledge.analysis import SyntaxVisitor

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI
    from cognition.self_model import FailureAnalysisEngine
class PatternFinderGene(Gene):
    """Encuentra patrones recurrentes en las estrategias exitosas."""
    def __init__(self, archive_to_analyze: str, intent_var: str, output_var: str, mind: 'PrometheusAGI', top_n: int = 3):
        self.archive_to_analyze = archive_to_analyze
        self.intent_var = intent_var
        self.output_var = output_var
        self.mind = mind
        self.top_n = top_n

    def execute(self, context: ExecutionContext):
        intent = context.get(self.intent_var)
        if self.archive_to_analyze == "specialists":
            chromosomes = self.mind.get_specialists_for_intent(intent) if intent else []
        else:
            chromosomes = self.mind.successful_strategies_archive

        if not chromosomes:
            context.set(self.output_var, "No encontré estrategias para analizar.")
            return
        
        sequences = [[g.__class__.__name__ for g in ch.genes] for ch in chromosomes]
        bigrams = Counter(bg for seq in sequences if len(seq) > 1 for bg in zip(seq, seq[1:]))
        trigrams = Counter(tg for seq in sequences if len(seq) > 2 for tg in zip(seq, seq[1:], seq[2:]))

        report = ["Patrones recurrentes en mis estrategias:"]
        if bigrams:
            report.append("\n--- Bigramas (2 Pasos) ---")
            for (g1, g2), count in bigrams.most_common(self.top_n):
                report.append(f"  - '{g1} -> {g2}' ({count} veces)")
        if trigrams:
            report.append("\n--- Trigramas (3 Pasos) ---")
            for (g1, g2, g3), count in trigrams.most_common(self.top_n):
                report.append(f"  - '{g1} -> {g2} -> {g3}' ({count} veces)")
        
        context.set(self.output_var, "\n".join(report))


class AnalyzeGeneSyntaxGene(Gene):
    """Analiza el código fuente de otro gen para entender su estructura."""
    def __init__(self, gene_name_var: str, output_var: str, gene_map: dict):
        self.gene_name_var = gene_name_var
        self.output_var = output_var
        self.gene_map = gene_map

    def _format_report(self, gene_name: str, analysis: dict) -> str:
        lines = [f"Análisis sintáctico de mi gen '{gene_name}':"]
        lines.append(f"  - Dependencias: {', '.join(analysis['imports'])}")
        lines.append(f"  - Lógica: {analysis['conditionals']} condicional(es), {analysis['loops']} bucle(s).")
        lines.append(f"  - Entradas (context.get): {', '.join(analysis['context_gets'])}")
        lines.append(f"  - Salidas (context.set): {', '.join(analysis['context_sets'])}")
        return "\n".join(lines)

    def execute(self, context: ExecutionContext):
        gene_name_raw = context.get("main_topic")
        if not gene_name_raw:
            context.set_final_response("No se especificó qué gen analizar.")
            return

        lower_gene_map = {k.lower(): v for k, v in self.gene_map.items()}
        target_class = lower_gene_map.get(gene_name_raw.lower())

        if not target_class:
            context.set_final_response(f"No tengo un gen llamado '{gene_name_raw}'.")
            return
        try:
            source_code = inspect.getsource(target_class)
            tree = ast.parse(source_code)
            visitor = SyntaxVisitor()
            visitor.visit(tree)
            report = self._format_report(target_class.__name__, visitor.analysis)
            context.set_final_response(report)
        except Exception as e:
            context.set_final_response(f"Error al analizar el gen '{target_class.__name__}': {e}")


class RecallLastResponseGene(Gene):
    """Recupera la última respuesta dada desde la memoria conversacional."""
    def __init__(self, mind: 'PrometheusAGI', output_var: str = "recalled_response"):
        self.mind = mind
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        if self.mind.conversational_history:
            last_response = self.mind.conversational_history[-1].get("prometheus")
            if last_response:
                context.set(self.output_var, f"Lo último que dije fue: '{last_response}'")
                return
        context.set(self.output_var, "No recuerdo haber dicho nada recientemente.")


class WhyDidIFailGene(Gene):
    """Activa el motor de análisis de fallos."""
    def __init__(self, failure_analyzer: 'FailureAnalysisEngine', output_var: str = "failure_report"):
        self.failure_analyzer = failure_analyzer
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        diagnosis = self.failure_analyzer.analyze_last_failure()
        context.set(self.output_var, diagnosis)