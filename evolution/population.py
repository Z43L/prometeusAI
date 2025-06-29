# prometheus_agi/evolution/population.py
import asyncio
import random
import copy
from typing import List, Dict, TYPE_CHECKING

# Importaciones del proyecto
from core.base import Chromosome, ExecutionContext, Gene
from evolution.engine import EvolutionEngine

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI

class PopulationGenerator:
    """Genera poblaciones de cromosomas para el Dojo."""
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind
        self.full_arsenal = list(mind.full_arsenal.values()) # Convertir dict a lista

    def generate(self, query: str, size: int) -> List[Chromosome]:
        """Genera una población inicial basada en la consulta."""
        print(f"\n[PopGen] Generando población para: '{query[:50]}...'")
        
        # En V5, la generación es más simple y no depende de un planificador complejo.
        # Se basa en especialistas y mutaciones aleatorias.
        analysis = self.mind.intent_engine.analyze(query)
        intent = analysis['intent']
        
        pop: List[Chromosome] = []
        specialists = self.mind.get_specialists_for_intent(intent)
        if specialists:
            pop.extend([copy.deepcopy(s) for s in specialists])
            print(f"  [PopGen] {len(specialists)} especialista(s) para '{intent}' añadidos como base.")

        # Añadimos el CognitiveCoreGene como estrategia fundamental
        cognitive_core_gene = self.mind.full_arsenal.get("CognitiveCoreGene")
        if cognitive_core_gene:
             pop.append(Chromosome([cognitive_core_gene], "Estrategia Cognitiva Central"))

        # Rellenamos el resto con mutaciones y caos
        temp_evolver = EvolutionEngine(self.full_arsenal, 0, {"swap": 0.5, "insertion": 0.5})
        while len(pop) < size:
            if pop:
                parent = max(pop, key=lambda c: c.fitness if c.fitness is not None else -1)
                child = temp_evolver._mutate(copy.deepcopy(parent))
                child.description += " (Hijo Mutado)"
                pop.append(child)
            else:
                # Fallback: crear un cromosoma aleatorio
                random_genes = random.choices(self.full_arsenal, k=random.randint(1, 3))
                pop.append(Chromosome(random_genes, "Hijo del Caos"))
                
        return pop[:size]

class ChromosomeExecutor:
    """Ejecuta los genes de un cromosoma en secuencia."""
    @staticmethod
    async def execute_async(chromosome: Chromosome, context: ExecutionContext):
        """Ejecuta la lógica de forma asíncrona para no bloquear."""
        print(f"[Executor] Ejecutando estrategia: '{chromosome.description}'")
        loop = asyncio.get_running_loop()
        for gene in chromosome.genes:
            print(f"  -> Próximo gen: {gene.__class__.__name__}")
            try:
                # Ejecuta el método síncrono en un hilo separado
                await loop.run_in_executor(None, gene.execute, context)
            except Exception as e:
                print(f"[ERROR FATAL EN EXECUTOR] El gen '{gene.__class__.__name__}' falló: {e}")
                import traceback
                traceback.print_exc()
                context.log_thought("FATAL_EXECUTOR_ERROR", f"Error en {gene.__class__.__name__}: {e}")
                break