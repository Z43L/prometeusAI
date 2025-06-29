# prometheus_agi/evolution/engine.py
import random
import copy
from typing import List, Dict, Tuple

# Importaciones del proyecto
from core.base import Chromosome, Gene

class EvolutionEngine:
    """Motor que aplica los operadores genéticos de mutación y cruce."""
    def __init__(self, arsenal: List[Gene], elite_size: int, mutation_config: Dict[str, float]):
        self.arsenal = arsenal
        self.elite_size = elite_size
        self.mutation_config = mutation_config
        
    def evolve(self, pop: List[Chromosome]) -> List[Chromosome]:
        """Ciclo de evolución principal."""
        pop.sort(key=lambda c: c.fitness, reverse=True)
        new_pop = pop[:self.elite_size]

        while len(new_pop) < len(pop):
            p1 = self._tournament_selection(pop)
            p2 = self._tournament_selection(pop)
            if not p1 or not p2: continue

            crossover_method = random.choice([self._crossover_single_point, self._crossover_two_point, self._crossover_uniform])
            child1, child2 = crossover_method(p1, p2)

            new_pop.append(self._mutate(child1))
            if len(new_pop) < len(pop):
                new_pop.append(self._mutate(child2))
        return new_pop

    def _tournament_selection(self, pop: List[Chromosome], k: int = 3) -> Chromosome:
        """Selecciona un padre de la población mediante torneo."""
        if not pop: return None
        selection = random.choices(pop, k=k)
        return max(selection, key=lambda c: c.fitness)

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Aplica una mutación aleatoria a un cromosoma."""
        mutation_type = random.choices(list(self.mutation_config.keys()), list(self.mutation_config.values()), k=1)[0]
        if mutation_type == 'swap': self._mutate_swap(chromosome)
        elif mutation_type == 'insertion': self._mutate_insertion(chromosome)
        elif mutation_type == 'deletion': self._mutate_deletion(chromosome)
        elif mutation_type == 'rearrangement': self._mutate_rearrangement(chromosome)
        elif mutation_type == 'parameter': self._mutate_parameter(chromosome)
        return chromosome

    # --- Operadores de Mutación ---
    def _mutate_swap(self, ch: Chromosome):
        if not ch.genes or not self.arsenal: return
        point = random.randint(0, len(ch.genes) - 1)
        ch.genes[point] = random.choice(self.arsenal)
        ch.description += " (Mut:Swap)"

    def _mutate_insertion(self, ch: Chromosome):
        if not self.arsenal: return
        point = random.randint(0, len(ch.genes))
        ch.genes.insert(point, random.choice(self.arsenal))
        ch.description += " (Mut:Ins)"

    def _mutate_deletion(self, ch: Chromosome):
        if len(ch.genes) > 1:
            ch.genes.pop(random.randint(0, len(ch.genes) - 1))
            ch.description += " (Mut:Del)"
            
    def _mutate_rearrangement(self, ch: Chromosome):
        if len(ch.genes) > 1:
            idx1, idx2 = random.sample(range(len(ch.genes)), 2)
            ch.genes[idx1], ch.genes[idx2] = ch.genes[idx2], ch.genes[idx1]
            ch.description += " (Mut:Rearr)"

    def _mutate_parameter(self, ch: Chromosome):
        if not ch.genes: return
        gene_to_mutate = random.choice(ch.genes)
        attrs = vars(gene_to_mutate)
        tunable_params = [k for k, v in attrs.items() if isinstance(v, (int, float))]
        if not tunable_params: return
        param_name = random.choice(tunable_params)
        current_value = getattr(gene_to_mutate, param_name)
        if isinstance(current_value, int):
            new_value = max(1, current_value + random.choice([-1, 1]))
        else: # float
            new_value = current_value * random.uniform(0.8, 1.2)
        setattr(gene_to_mutate, param_name, new_value)
        ch.description += f" (Mut:Param)"

    # --- Operadores de Cruce ---
    def _crossover_single_point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        if not p1.genes or not p2.genes: return copy.deepcopy(p1), copy.deepcopy(p2)
        point = random.randint(1, min(len(p1.genes), len(p2.genes))) if min(len(p1.genes), len(p2.genes)) > 1 else 1
        c1_genes = p1.genes[:point] + p2.genes[point:]
        c2_genes = p2.genes[:point] + p1.genes[point:]
        return Chromosome(c1_genes, "Hijo de Cruce 1P"), Chromosome(c2_genes, "Hijo de Cruce 1P")

    def _crossover_two_point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        size = min(len(p1.genes), len(p2.genes))
        if size < 2: return self._crossover_single_point(p1, p2)
        pt1, pt2 = sorted(random.sample(range(size), 2))
        c1_genes = p1.genes[:pt1] + p2.genes[pt1:pt2] + p1.genes[pt2:]
        c2_genes = p2.genes[:pt1] + p1.genes[pt1:pt2] + p2.genes[pt2:]
        return Chromosome(c1_genes, "Hijo de Cruce 2P"), Chromosome(c2_genes, "Hijo de Cruce 2P")

    def _crossover_uniform(self, p1: Chromosome, p2: Chromosome, swap_prob: float = 0.5) -> Tuple[Chromosome, Chromosome]:
        c1_genes, c2_genes = [], []
        len_min = min(len(p1.genes), len(p2.genes))
        for i in range(len_min):
            if random.random() < swap_prob:
                c1_genes.append(p2.genes[i])
                c2_genes.append(p1.genes[i])
            else:
                c1_genes.append(p1.genes[i])
                c2_genes.append(p2.genes[i])
        c1_genes.extend(p1.genes[len_min:])
        c2_genes.extend(p2.genes[len_min:])
        return Chromosome(c1_genes, "Hijo de Cruce Uni"), Chromosome(c2_genes, "Hijo de Cruce Uni")