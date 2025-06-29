# prometheus_agi/evolution/dojo.py
import gc
import time
import copy
import random
import multiprocessing
from typing import List, Dict, Tuple, TYPE_CHECKING
import os
import queue

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI

# Importaciones del proyecto
from config import (
    BATCH_SIZE, EVOLUTION_FOCUS_SIZE, PROMETHEUS_CANDIDATE_PERCENTILE,
    STAGNATION_LIMIT, INCAPACITY_THRESHOLD, INCAPACITY_ATTEMPTS
)
from core.base import Chromosome, ExecutionContext
from evolution.population import PopulationGenerator, ChromosomeExecutor
from evolution.fitness import FitnessCalculator
from evolution.engine import EvolutionEngine
from cognition.self_model import MetacognitionEngine
from environments.grid_world import WorldStream
# Dependencias que deben estar en el scope global para los workers de multiprocessing
from sentence_transformers import SentenceTransformer
import spacy
from knowledge.graph import KnowledgeGraph

# --- Funciones para Workers de Multiprocessing ---

def init_worker(model_name: str, spacy_model_name: str, neo4j_uri: str, neo4j_user: str, neo4j_pass: str):
    """Inicializa cada proceso del pool con los recursos necesarios."""
    global worker_model, worker_nlp, worker_kg
    print(f"[Worker-Init pid={os.getpid()}] Iniciando worker...")
    worker_model = SentenceTransformer(model_name)
    worker_nlp = spacy.load(spacy_model_name)
    worker_kg = KnowledgeGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    print(f"[Worker-Init pid={os.getpid()}] Worker listo.")

def _worker_evaluate_chromosome_base(args: Tuple):
    """La tarea que ejecuta un worker: evaluar un cromosoma."""
    chromosome, ctx_vars, ideal_response, gene_usage_counter = args
    # El executor es síncrono dentro del worker
    for gene in chromosome.genes:
        gene.execute(ctx_vars)
    chromosome.final_context = ctx_vars
    
    # El calculator usa el modelo cargado en el worker
    calculator = FitnessCalculator(sim_model=worker_model)
    raw_scores = calculator.calculate_metrics(chromosome, ideal_response, gene_usage_counter)
    return chromosome, raw_scores

# --- Clase Principal del Dojo ---

class Dojo:
    """
    Orquesta el entrenamiento completo, la evolución y la exploración autónoma de Prometheus.
    Es el núcleo del sistema de aprendizaje y auto-mejora.
    """
    def __init__(self, prometheus: 'PrometheusAGI'):
        self.prometheus = prometheus
        if not self.prometheus.training_corpus and not prometheus.force_genesis:
             print("[WARN] No se encontró un corpus de entrenamiento. El modo 'Maratón' no funcionará.")
        
        print("[DOJO] Configurando herramientas evolutivas, de NLU y metacognitivas...")
        
        self.population_generator = self.prometheus.population_generator
        self.fitness_calculator = FitnessCalculator(sim_model=prometheus.similarity_model)
        self.metacognition_engine = MetacognitionEngine(prometheus_mind=prometheus)
        
        mutation_probabilities = {
            "swap": 0.3, "insertion": 0.2, "deletion": 0.2,
            "rearrangement": 0.2, "parameter": 0.1
        }
        
        # --- CAMBIO IMPORTANTE AQUÍ ---
        # El motor de evolución ahora usa el arsenal seguro y serializable
        self.evolution_engine = EvolutionEngine(
            arsenal=self.prometheus.dojo_picklable_arsenal, 
            elite_size=2,
            mutation_config=mutation_probabilities,
        )
        # -----------------------------
        
        self.fitness_cache = {}
        self.STAGNATION_LIMIT = 4
        self.incapacity_tracker = {}
        self.INCAPACITY_THRESHOLD = 300
        self.INCAPACITY_ATTEMPTS = 3


    def _get_chromosome_signature(self, chromosome: 'Chromosome') -> tuple:
        """Crea una representación hashable de un cromosoma para usarla como clave de caché."""
        sig = []
        for gene in chromosome.genes:
            params = tuple(sorted((k, v) for k, v in vars(gene).items() if isinstance(v, (str, int, float, bool))))
            sig.append((gene.__class__.__name__, params))
        return tuple(sig)

    def train_on_last_episode(self, generations: int = 5, population_size: int = 10):
        """
        Toma el último episodio de la memoria, lo convierte en un desafío y
        ejecuta un ciclo de evolución para mejorar la estrategia usada.
        """
        print("\n===== MODO DE REFLEXIÓN POST-EPISODIO (DOJO) =====")
        goal_events = self.prometheus.episodic_memory.get_events_by_type("GOAL_COMPLETED", limit=1)
        if not goal_events:
            print("[DOJO_REFLEX] No hay objetivos completados recientemente para analizar.")
            return

        last_goal = goal_events[0]['details']
        instruction = last_goal['description']
        
        ideal_response = f"Objetivo '{instruction}' completado."
        intent = "ACTUAR_EN_ENTORNO"

        best_chromosome = self._evolve_on_challenge(
            instruction=instruction,
            ideal_response=ideal_response,
            intent=intent,
            generations=generations,
            population_size=population_size
        )

        if best_chromosome:
            print(f"[DOJO_REFLEX] Reflexión completada. Nueva estrategia especialista encontrada con Fitness: {best_chromosome.fitness:.2f}")
            self.prometheus.update_specialist(intent, best_chromosome)
        
        print("====================== REFLEXIÓN FINALIZADA ======================\n")

    # EN: la clase Dojo
    # REEMPLAZA el método _evolve_on_challenge completo con esta versión corregida:

    def _evolve_on_challenge(self, instruction: str, ideal_response: str, intent: str, generations: int, population_size: int) -> 'Chromosome':
        print(f"  [EVOLVING] On: '{instruction[:60]}...' for intent '{intent}'")
        
        # ... (código de análisis de intención y generación de población sin cambios) ...
        analysis = self.prometheus.intent_engine.analyze(instruction)
        base_ctx_vars = {"query": instruction, "intent": analysis['intent'], "main_topic": analysis['main_topic']}
        population = self.population_generator.generate(instruction, population_size)
        best_overall_chromosome = None
        last_best_fitness = -float('inf')
        stagnation_counter = 0
        PROMETHEUS_CANDIDATE_PERCENTILE = 0.4 

        for gen in range(generations):
            fitness_profile = self.prometheus.get_profile_for_intent(intent)
            print(f"    Gen {gen+1}/{generations} | Etapa 1: Evaluación ligera para {len(population)} cromosomas.")
            
            # ... (Etapa 1: Evaluación ligera sin cambios) ...
            cheap_scores = {}
            for i, chromosome in enumerate(population):
                signature = self._get_chromosome_signature(chromosome)
                raw_scores = {"efficiency": -len(chromosome.genes), "curiosity": sum(1.0 - (self.prometheus.gene_usage_counter.get(g.__class__.__name__, 0) / (sum(self.prometheus.gene_usage_counter.values()) or 1)) for g in chromosome.genes), "novelty": self.fitness_calculator.calculate_structural_novelty(chromosome, population[:i] + population[i+1:])}
                preliminary_fitness = (raw_scores["efficiency"] * fitness_profile.weights["efficiency"] + raw_scores["curiosity"] * fitness_profile.weights["curiosity"] + raw_scores["novelty"] * fitness_profile.weights["novelty"])
                cheap_scores[signature] = {"raw": raw_scores, "preliminary_fitness": preliminary_fitness}
            
            population.sort(key=lambda c: cheap_scores[self._get_chromosome_signature(c)]["preliminary_fitness"], reverse=True)
            
            # ... (Etapa 2: Selección de candidatos sin cambios) ...
            num_promising_candidates = int(len(population) * PROMETHEUS_CANDIDATE_PERCENTILE)
            promising_candidates = population[:num_promising_candidates]
            less_promising_population = population[num_promising_candidates:]
            
            print(f"    Etapa 2: Seleccionados {len(promising_candidates)} candidatos para evaluación intensiva.")
            
            tasks_to_run_args = []
            cached_chromosomes = []
            for chromosome in promising_candidates:
                signature = self._get_chromosome_signature(chromosome)
                if signature in self.fitness_cache:
                    chromosome.fitness = self.fitness_cache[signature]['fitness']
                    cached_chromosomes.append(chromosome)
                else:
                    tasks_to_run_args.append((chromosome, base_ctx_vars, ideal_response, self.prometheus.gene_usage_counter))

            print(f"    Etapa 3 | Caché: {len(cached_chromosomes)} hits | A evaluar intensivamente: {len(tasks_to_run_args)} tasks")
            
            newly_evaluated_chromosomes = []
            if tasks_to_run_args:
                results = []
                num_processes = 2
                print(f"    [!] Limitando el pool a {num_processes} procesos para conservar memoria.")
                pool_initializer = init_worker
                pool_init_args = (self.prometheus.similarity_model_name, "es_core_news_sm", self.prometheus.knowledge_graph.uri, self.prometheus.knowledge_graph.user, self.prometheus.knowledge_graph.password)

                try:
                    with multiprocessing.Pool(processes=num_processes, initializer=pool_initializer, initargs=pool_init_args) as pool:
                        results = pool.map(_worker_evaluate_chromosome_base, tasks_to_run_args)
                except Exception as e:
                    print(f"[ERROR] Falló el pool de multiprocessing: {e}")
                    break
                
                for chromosome, heavy_raw_scores in results:
                    signature = self._get_chromosome_signature(chromosome)
                    full_raw_scores = {**cheap_scores.get(signature, {"raw": {}})["raw"], **heavy_raw_scores}
                    chromosome.fitness = fitness_profile.calculate_total_fitness(full_raw_scores)
                    self.fitness_cache[signature] = {"fitness": chromosome.fitness, "raw": full_raw_scores}
                    newly_evaluated_chromosomes.append(chromosome)
                    
                    # --- NUEVO: GESTIÓN DE CACHÉ Y MEMORIA ---
                    if len(self.fitness_cache) > 5000: # Límite de caché más estricto
                        keys_to_delete = list(self.fitness_cache.keys())[:2500]
                        for key in keys_to_delete:
                            del self.fitness_cache[key]
                        print("    [!] Se ha limpiado la mitad del fitness_cache para liberar memoria.")
                    
                    # Liberamos explícitamente la memoria del contexto que ya no se necesita
                    del chromosome.final_context
                    # --- FIN DEL BLOQUE NUEVO ---

            # ... (Etapa 4: Combinar poblaciones y lógica de estancamiento sin cambios) ...
            for chromosome in less_promising_population:
                signature = self._get_chromosome_signature(chromosome)
                if signature in cheap_scores:
                    chromosome.fitness = cheap_scores[signature]["preliminary_fitness"]

            population = cached_chromosomes + newly_evaluated_chromosomes + less_promising_population
            population.sort(key=lambda c: c.fitness, reverse=True)
            current_best_fitness = population[0].fitness if population else -float('inf')
            print(f"    Etapa 4: Mejor Fitness de la Gen: {current_best_fitness:.2f} | Cromosomas totales: {len(population)}")

            if best_overall_chromosome is None or current_best_fitness > best_overall_chromosome.fitness:
                best_overall_chromosome = copy.deepcopy(population[0]) if population else None
            
            if current_best_fitness <= last_best_fitness: stagnation_counter += 1
            else:
                last_best_fitness = current_best_fitness
                stagnation_counter = 0

            if stagnation_counter >= self.STAGNATION_LIMIT:
                print("    [!] Estancamiento detectado. Deteniendo evolución para este desafío.")
                break
                
            population = self.evolution_engine.evolve(population)
            
            # --- NUEVO: FORZAR LA RECOLECCIÓN DE BASURA ---
            # Al final de cada generación, le decimos a Python que limpie la memoria no utilizada.
            gc.collect()
            # ----------------------------------------------
        
            if best_overall_chromosome:
                print(f"  [EVOLVED] Estrategia optimizada encontrada: '{best_overall_chromosome.description}' (Fitness Final: {best_overall_chromosome.fitness:.2f})")
                if best_overall_chromosome.fitness > 500:
                    self.prometheus.update_specialist(intent, best_overall_chromosome)
                if intent in self.incapacity_tracker:
                    self.incapacity_tracker[intent]['scores'].append(best_overall_chromosome.fitness or 0)
        
        return best_overall_chromosome



    def run_marathon(self, generations: int, population_size: int):
        """
        Ejecuta el entrenamiento supervisado en modo 'Maratonista' usando el dataset.
        """
        if not self.prometheus.training_corpus:
            print("[DOJO] No hay corpus de entrenamiento. Saltando modo Maratón.")
            return

        print("\n" + "="*25 + " INICIANDO ENTRENAMIENTO DEL DOJO: MODO MARATONISTA " + "="*25)
        corpus = self.prometheus.training_corpus
        num_batches = 5
        
        for i in range(num_batches):
            print(f"\n--- LOTE {i+1}/{num_batches} ---")
            batch_start = i * BATCH_SIZE
            batch_end = min((i + 1) * BATCH_SIZE, len(corpus))
            current_batch = corpus[batch_start:batch_end]
            
            batch_results = []
            for challenge in current_batch:
                analysis = self.population_generator.intent_engine.analyze(challenge["INSTRUCTION"])
                # Llama a think del Prometheus principal, que ya no usará LLM para la respuesta final.
                # No necesitamos ground_truth_response aquí, solo en el _evolve_on_challenge
                # Por simplicidad, se mantiene el attempt que usa el archive
                _, score, _ = self.prometheus.attempt(challenge["INSTRUCTION"], challenge["RESPONSE"])
                batch_results.append({
                    "instruction": challenge["INSTRUCTION"], "ideal_response": challenge["RESPONSE"],
                    "score": score, "intent": analysis['intent']
                })

            challenges_to_evolve = sorted(batch_results, key=lambda x: x["score"])[:EVOLUTION_FOCUS_SIZE]
            
            for challenge in challenges_to_evolve:
                self.incapacity_tracker.setdefault(challenge['intent'], {'attempts': 0, 'scores': []})
                self.incapacity_tracker[challenge['intent']]['attempts'] += 1
                
                best_chromosome = self._evolve_on_challenge(
                    instruction=challenge["instruction"],
                    ideal_response=challenge["ideal_response"],
                    intent=challenge["intent"],
                    generations=generations,
                    population_size=population_size
                )                
                if best_chromosome:
                     self.prometheus.archive_strategy(best_chromosome)
                     self.prometheus.record_gene_usage(best_chromosome)

            self.metacognition_engine.analyze_and_evolve_profiles(batch_results)

            for intent, data in self.incapacity_tracker.items():
                if data['attempts'] >= self.INCAPACITY_ATTEMPTS:
                    avg_score = sum(data['scores']) / len(data['scores']) if data['scores'] else 0
                    if avg_score < self.INCAPACITY_THRESHOLD:
                        print(f"[DOJO] ¡Incapacidad Crónica detectada para la intención '{intent}'!")
                        sample_instruction = next(c['instruction'] for c in batch_results if c['intent'] == intent)
                        self.prometheus.gene_forge.attempt_new_gene_creation(intent, sample_instruction)
                    self.incapacity_tracker[intent] = {'attempts': 0, 'scores': []}

            self.prometheus._save_mind_state()
            
        print("\n" + "="*35 + " ENTRENAMIENTO MARATÓN COMPLETADO " + "="*35)
    
    
    def run_exploratory_mode(self, num_episodes: int, generations_per_episode: int, population_size: int = 20):
        """
        Ejecuta un modo de entrenamiento no supervisado donde la IA explora por curiosidad intelectual,
        intentando resolver desafíos conocidos de formas nuevas y creativas.
        """
        print("\n" + "="*25 + " INICIANDO MODO EXPLORATORIO " + "="*25)
        
        if not self.prometheus.training_corpus:
            print("[DOJO] No hay corpus de entrenamiento para explorar. Saltando modo Exploratorio.")
            return

        for i in range(num_episodes):
            print(f"\n--- Episodio de Exploración {i+1}/{num_episodes} ---")
            
            # 1. La IA genera su propio objetivo eligiendo un desafío al azar de su curriculum
            challenge = random.choice(self.prometheus.training_corpus)
            instruction = challenge["INSTRUCTION"]
            
            print(f"  [Exploratory Goal] Mi objetivo auto-generado es: 'Explorar nuevas soluciones para: {instruction[:50]}...'")

            # 2. La evolución optimiza para ese objetivo interno usando el perfil EXPLORAR.
            #    La respuesta ideal se deja vacía para no limitar la creatividad.
            best_chromosome = self._evolve_on_challenge(
                instruction=instruction,
                ideal_response="", # No hay respuesta "correcta", se busca novedad.
                intent="EXPLORAR", # <<< USA EL PERFIL DE FITNESS EXPLORATORIO
                generations=generations_per_episode,
                population_size=population_size
            )
            
            if best_chromosome:
                self.prometheus.record_gene_usage(best_chromosome)
                self.prometheus.archive_strategy(best_chromosome)
            
            self.prometheus._save_mind_state()
            time.sleep(1)
            
        print("\n" + "="*35 + " MODO EXPLORATORIO COMPLETADO " + "="*35)


class ConsciousnessLoop:
    """
    El bucle principal de la "vida" de Prometheus. Escucha el flujo de percepciones
    y decide cuándo activar el proceso de pensamiento de la IA.
    """
    def __init__(self, prometheus_instance: 'PrometheusAGI', world_stream: 'WorldStream'):
        self.prometheus = prometheus_instance
        self.world_stream = world_stream

    def run(self):
        print("[CONSCIOUSNESS] El ciclo de conciencia ha comenzado. Escuchando al mundo...")
        
        while True:
            try:
                percept = self.world_stream.perception_queue.get(timeout=10)
                
                print(f"\n[CONSCIOUSNESS] Nueva percepción recibida: {percept.type} a las {percept.timestamp:.2f}")
                
                if percept.type in ["observation", "termination"]:
                    action_chromosome = self.prometheus.think_and_act(percept.data['obs'])
                    
                    print(f"[DEBUG] Contexto final de la Mente: {action_chromosome.final_context.memory}")

                    action = action_chromosome.final_context.get("chosen_action")
                    
                    if action is not None:
                        print(f"[CONSCIOUSNESS] Decisión tomada. Enviando acción {action} al mundo.")
                        self.world_stream.action_queue.put(action)

                    self.prometheus.goal_manager.update_goals_status(percept.data['obs'])
                    goal_completed = self.prometheus.goal_manager.update_goals_status(percept.data['obs'])
                    if goal_completed:
                        dojo = Dojo(self.prometheus)
                        dojo.train_on_last_episode()
                if percept.type == "termination":
                    pass

            except queue.Empty:
                print("[CONSCIOUSNESS] No se han recibido percepciones del mundo. El sistema está inactivo.")
            except KeyboardInterrupt:
                print("[CONSCIOUSNESS] Interrupción manual. Deteniendo el ciclo de conciencia.")
                break
                
        print("[CONSCIOUSNESS] Ciclo de conciencia terminado.")
