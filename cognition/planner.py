# prometheus_agi/cognition/planner.py
# En la V5, el StrategicPlanner no está completamente implementado como en versiones posteriores.
# El pensamiento se centraliza en el CognitiveCoreGene.
# Mantenemos la clase para coherencia estructural.

from typing import List, Dict
from core.base import Chromosome, Gene, GeneDescriptor

class StrategicPlanner:
    """
    Decide qué genes usar. En V5, esta lógica es simple y está mayormente
    implícita en el CognitiveCoreGene.
    """
    def __init__(self, gene_descriptors: List[GeneDescriptor], full_arsenal: Dict[str, Gene], similarity_model, self_model):
        self.gene_descriptors = gene_descriptors
        self.gene_map = {gene.__class__: gene for gene in full_arsenal.values()}
        self.similarity_model = similarity_model
        self.self_model = self_model
        print("[PLANNER] Planificador Estratégico (V5 - simplificado) inicializado.")

    def plan_seed_strategy(self, query: str, analysis: Dict) -> Chromosome:
        """
        Genera una estrategia semilla. En V5, esto es un placeholder ya que
        la lógica principal está en CognitiveCoreGene.
        """
        # En esta versión, simplemente devuelve una estrategia vacía o una por defecto,
        # ya que el `think` de PrometheusAGI no lo utiliza activamente.
        from genes.system import CognitiveCoreGene
        
        # Obtenemos la instancia del gen del mapa de genes
        cognitive_core_gene_instance = self.gene_map.get(CognitiveCoreGene)

        if cognitive_core_gene_instance:
            return Chromosome(
                [cognitive_core_gene_instance],
                "Estrategia Cognitiva Central (Plan V5)"
            )
        else:
            # Fallback muy improbable
            return Chromosome([], "Estrategia vacía")