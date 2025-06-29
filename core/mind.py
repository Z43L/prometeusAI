# prometheus_agi/core/mind.py
import os
import json
import pickle
import spacy
import asyncio
from collections import Counter
from typing import Dict, Any, List

from sentence_transformers import SentenceTransformer

# Importaciones de la nueva estructura de proyecto
from config import (
    PATHS, MIND_STATE_DIR, SIMILARITY_MODEL_NAME, SPACY_MODEL_NAME, MAX_SPECIALISTS_PER_INTENT
)
from core.base import Chromosome, Gene, GeneDescriptor, ExecutionContext
from knowledge.graph import KnowledgeGraph
from knowledge.memory import EpisodicMemory
from cognition.intent import DynamicIntentEngine
from cognition.self_model import SelfModel, FailureAnalysisEngine
from cognition.goal_manager import GoalManager
from evolution.dojo import Dojo
from evolution.population import ChromosomeExecutor, PopulationGenerator
from evolution.fitness import FitnessProfile

# Importaciones de Genes
from genes.tools import WebSearchGene, CalculatorGene, ScientificSearchGene
from genes.system import (
    CognitiveCoreGene, GetNodeDefinitionGene, LearnFromTextGene, DeepReasoningGene,
    LinguisticAnalysisGene, InferDefinitionFromNeighborsGene, FormulateResponseGene
)
from genes.introspective import RecallLastResponseGene, WhyDidIFailGene
from genes.environment import PerceiveEnvironmentGene, CheckGoalStatusGene, CuriosityGene, DecideActionGene
from genes.interaction import DetectSentimentGene, StyleAnalyzerGene

class PrometheusAGI:
    """
    La clase principal que representa la mente de la AGI.
    """
    def __init__(self, force_genesis: bool = False, **kwargs):
        print("[MIND] Iniciando Prometheus...")
        self.force_genesis = force_genesis
        self.paths = PATHS
        self.conversational_history: List[Dict[str, str]] = []
        self.successful_strategies_archive: List[Chromosome] = []
        self.intent_specialists: Dict[str, List[Chromosome]] = {}
        self.fitness_profiles_by_intent: Dict[str, FitnessProfile] = {}
        self.gene_usage_counter = Counter()
        self.training_corpus: List[Dict[str, str]] = []
        self.env_size = 10

        os.makedirs(MIND_STATE_DIR, exist_ok=True)

        print("[MIND] Cargando componentes base...")
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.create_vector_index()
        self.episodic_memory = EpisodicMemory()
        self.similarity_model_name = SIMILARITY_MODEL_NAME
        self._similarity_model = None

        try:
            self.lightweight_nlp = spacy.load(SPACY_MODEL_NAME)
        except OSError:
            print(f"ADVERTENCIA: '{SPACY_MODEL_NAME}' no encontrado. Ejecuta: python -m spacy download {SPACY_MODEL_NAME}")
            exit()

        self.intent_engine = DynamicIntentEngine()
        self.failure_analyzer = FailureAnalysisEngine(self)

        print("[MIND] Creando e inyectando genoma...")
        self.tool_genes = self._create_tool_genes()
        self.full_arsenal = self._create_full_arsenal()
        self.tool_descriptors = self._create_tool_descriptors()

        self._load_mind_state()
        
        # Initialize population generator before dojo
        self.population_generator = PopulationGenerator(self)
        
        # Create picklable arsenal for dojo evolution engine
        self.dojo_picklable_arsenal = [
            WebSearchGene(),
            CalculatorGene(),
            ScientificSearchGene(),
            DetectSentimentGene(),
            FormulateResponseGene(),
            StyleAnalyzerGene(),
            PerceiveEnvironmentGene()
        ]
        
        self.dojo = Dojo(self)

    def _create_tool_genes(self) -> Dict[str, Gene]:
        """Crea instancias de los genes que son herramientas."""
        return {
            "WebSearchGene": WebSearchGene(),
            "CalculatorGene": CalculatorGene(),
            "ScientificSearchGene": ScientificSearchGene(),
        }

    def _create_full_arsenal(self) -> Dict[str, Gene]:
        """Crea el genoma completo inyectando las dependencias necesarias."""
        return {
            # --- GENES DE SISTEMA ---
            "CognitiveCoreGene": CognitiveCoreGene(mind=self), # Necesita la mente para coordinar
            "GetNodeDefinitionGene": GetNodeDefinitionGene(graph=self.knowledge_graph),
            "LearnFromTextGene": LearnFromTextGene(graph=self.knowledge_graph, nlp_processor=self.lightweight_nlp),
            "DeepReasoningGene": DeepReasoningGene(graph=self.knowledge_graph, nlp_processor=self.lightweight_nlp),
            "LinguisticAnalysisGene": LinguisticAnalysisGene(nlp_processor=self.lightweight_nlp),
            "InferDefinitionFromNeighborsGene": InferDefinitionFromNeighborsGene(graph=self.knowledge_graph),
            "FormulateResponseGene": FormulateResponseGene(),

            # --- GENES DE ENTORNO ---
            "PerceiveEnvironmentGene": PerceiveEnvironmentGene(),
            "CheckGoalStatusGene": CheckGoalStatusGene(goal_manager=GoalManager(self)), # Inyectar GoalManager
            "CuriosityGene": CuriosityGene(mind=self),
            "DecideActionGene": DecideActionGene(),

            # --- GENES INTROSPECTIVOS ---
            "RecallLastResponseGene": RecallLastResponseGene(mind=self),
            "WhyDidIFailGene": WhyDidIFailGene(failure_analyzer=self.failure_analyzer),
        }

    def _create_tool_descriptors(self) -> List[GeneDescriptor]:
        """Crea metadatos para las herramientas seleccionables."""
        return [
            GeneDescriptor(WebSearchGene, 'buscar en internet', input_variables=['query'], output_variable='web_summary'),
            GeneDescriptor(CalculatorGene, 'resolver una operación matemática', input_variables=['query'], output_variable='calculation_result'),
            GeneDescriptor(ScientificSearchGene, 'buscar artículos científicos', input_variables=['query'], output_variable='scientific_summary'),
        ]

    @property
    def similarity_model(self):
        if self._similarity_model is None:
            print(f"\n[MIND] Cargando modelo de similitud '{self.similarity_model_name}'...")
            self._similarity_model = SentenceTransformer(self.similarity_model_name)
        return self._similarity_model

    async def think(self, query: str) -> str:
        print(f"\n[MIND] Procesando: '{query}'...")
        cognitive_strategy = Chromosome(genes=[self.full_arsenal["CognitiveCoreGene"]], description="Estrategia Cognitiva Central")
        final_context = ExecutionContext()
        final_context.set('query', query)  # Set the query in the context memory
        await ChromosomeExecutor.execute_async(cognitive_strategy, final_context)
        final_response = final_context.get_final_response() or "No he podido formular una respuesta."
        self.conversational_history.append({"user": query, "prometheus": final_response})
        return final_response
    
    # ... (El resto de los métodos de PrometheusAGI como shutdown, load/save_state, etc. permanecen igual)
    def shutdown(self):
        print("\n[MIND] Apagando...")
        self.knowledge_graph.close()
        self._save_mind_state()
        print("[MIND] Apagado completado.")

    def _load_mind_state(self):
        print("[MIND] Comprobando estado mental guardado...")
        if self.force_genesis: return
        
        files_to_load = {
            "episodic_memory": "episodic_memory",
            "general_archive": "successful_strategies_archive",
            "specialists_archive": "intent_specialists",
            "fitness_profiles": "fitness_profiles_by_intent",
            "gene_usage": "gene_usage_counter"
        }
        for path_key, attr_name in files_to_load.items():
            file_path = self.paths.get(path_key)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, "rb") as f:
                        setattr(self, attr_name, pickle.load(f))
                except (pickle.UnpicklingError, EOFError):
                     print(f"[WARN] El archivo '{file_path}' está corrupto o vacío. Se ignorará.")
        
        dojo_path = self.paths.get('dojo_dataset')
        if dojo_path and os.path.exists(dojo_path):
             with open(dojo_path, 'r', encoding='utf-8') as f:
                self.training_corpus = json.load(f)

    def _save_mind_state(self):
        print("[MIND] Guardando estado mental...")
        data_to_save = {
            "episodic_memory": self.episodic_memory,
            "successful_strategies_archive": self.successful_strategies_archive,
            "intent_specialists": self.intent_specialists,
            "fitness_profiles_by_intent": self.fitness_profiles_by_intent,
            "gene_usage_counter": self.gene_usage_counter,
        }
        for attr_name, data in data_to_save.items():
            # Construye la ruta del archivo a partir del nombre del atributo
            file_path = os.path.join(MIND_STATE_DIR, f"{attr_name}.pkl")
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
            except Exception as e:
                print(f"[ERROR] No se pudo guardar '{file_path}': {e}")