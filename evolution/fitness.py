# prometheus_agi/evolution/fitness.py
import random
import torch
from typing import Dict, List
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Importaciones del proyecto
from core.base import Chromosome

class FitnessProfile:
    """Representa los criterios de juicio para una intención específica."""
    def __init__(self, intent_name: str):
        self.intent_name = intent_name
        self.weights = {
            "similarity": 1.0, "efficiency": -10.0, "detail": 0.01,
            "novelty": 0.1, "curiosity": 0.0, "empowerment": 0.0,
        }
        if self.intent_name in ["EXPLORAR", "AUTO_MEJORA"]:
            self.weights.update({
                "similarity": 0.0, "curiosity": 50.0, "empowerment": 1000.0,
                "efficiency": -25.0, "novelty": 5.0
            })

    def calculate_total_fitness(self, scores: Dict[str, float]) -> float:
        """Calcula el fitness total aplicando los pesos a las puntuaciones."""
        return sum(scores.get(key, 0.0) * weight for key, weight in self.weights.items())

    def mutate(self, learning_rate: float = 0.1):
        """Muta aleatoriamente uno de los pesos para evolucionar el criterio."""
        param_to_mutate = random.choice(list(self.weights.keys()))
        change_factor = random.uniform(-learning_rate, learning_rate)
        if self.weights[param_to_mutate] >= 0:
            self.weights[param_to_mutate] = max(0, self.weights[param_to_mutate] * (1 + change_factor))
        else:
            self.weights[param_to_mutate] *= (1 + change_factor)
        print(f"  [MUTATE_PROFILE] Perfil '{self.intent_name}' mutado. Nuevo peso para '{param_to_mutate}': {self.weights[param_to_mutate]:.2f}")

    def __repr__(self):
        return f"<FitnessProfile({self.intent_name})>"

class FitnessCalculator:
    """Calcula un conjunto de métricas de rendimiento para un cromosoma."""
    def __init__(self, sim_model: SentenceTransformer):
        self.model = sim_model

    def calculate_metrics(self, chromosome: Chromosome, ideal_response: str, gene_usage: Counter) -> Dict[str, float]:
        """Calcula las métricas que no dependen de otras respuestas de la población."""
        if not chromosome.final_context or chromosome.final_context.get("execution_error"):
            return {"similarity": 0, "efficiency": -100, "detail": 0, "curiosity": 0}

        actual_response = chromosome.final_context.get_final_response()
        if not actual_response:
            return {"similarity": 0, "efficiency": -len(chromosome.genes), "detail": 0, "curiosity": 0}

        # Similitud
        similarity_score = 0.0
        if ideal_response:
            try:
                emb_p = self.model.encode(actual_response, convert_to_tensor=True)
                emb_m = self.model.encode(ideal_response, convert_to_tensor=True)
                similarity_score = util.pytorch_cos_sim(emb_p, emb_m).item()
            except Exception:
                pass
        
        # Curiosidad
        curiosity_score = 0.0
        total_uses = sum(gene_usage.values()) or 1
        for gene in chromosome.genes:
            rarity = 1.0 - (gene_usage.get(gene.__class__.__name__, 0) / total_uses)
            curiosity_score += rarity
            
        return {
            "similarity": similarity_score * 1000,
            "efficiency": -len(chromosome.genes),
            "detail": len(actual_response),
            "curiosity": curiosity_score,
        }

    def calculate_structural_novelty(self, chromosome: Chromosome, population_chromosomes: List[Chromosome]) -> float:
        """Calcula la novedad estructural de un cromosoma frente a la población."""
        if not population_chromosomes:
            return 1.0
        current_gene_set = set(g.__class__.__name__ for g in chromosome.genes)
        total_jaccard_distance = 0
        for other_chromosome in population_chromosomes:
            if other_chromosome == chromosome: continue
            other_gene_set = set(g.__class__.__name__ for g in other_chromosome.genes)
            intersection = len(current_gene_set.intersection(other_gene_set))
            union = len(current_gene_set.union(other_gene_set))
            jaccard_similarity = intersection / union if union > 0 else 1.0
            total_jaccard_distance += (1.0 - jaccard_similarity)
        return total_jaccard_distance / len(population_chromosomes) if population_chromosomes else 1.0