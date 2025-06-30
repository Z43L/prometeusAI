# prometheus_v4.py
# El Genoma Completo de una Inteligencia General Evolutiva y Explicable
# Creador: David
# Mentor: El propio Dataset
import gc
import asyncio
import sys
from neo4j import GraphDatabase, basic_auth
from dataclasses import dataclass, field
import inspect
import multiprocessing
import abc
import json
import aiohttp
import os
import random
import copy
import re
import time
from typing import Dict, Any, List, Tuple, Iterable
from collections import Counter
import requests
import pickle
import argparse
import networkx as nx
from multiprocessing import Process, Queue, Event
from transformers import pipeline
import spacy
import operator
import torch
import importlib.util
import ast
import operator as op
GOOGLE_API_KEY = "AIzaSyC8gKreBW0CNtEYcvfU5FH3g_Q-LJURpq8"
GOOGLE_CX = "22b0b9b4060f94ccc"
# --- Dependencias de Terceros ---
print("Cargando dependencias... Puede tardar unos segundos.")
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
import pyttsx3
import speech_recognition as sr
import google.generativeai as genai
#from environment import GridWorldEnv # Importar el entorno que creamos
print("[NLTK] Verificando paquetes de datos de lenguaje...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) # <<< AÑADE ESTA LÍNEA
print("[NLTK] Verificación completada.")
# Ejecuta esto una vez en un script de Python o en una consola interactiva

# --- ADVERTENCIA DE SEGURIDAD ---
# NO GUARDES TUS CLAVES DE API DIRECTAMENTE EN EL CÓDIGO.
# Utiliza variables de entorno. Antes de ejecutar, en tu terminal:
# En Windows (CMD): set GOOGLE_API_KEY="AIza..."
# En Linux/macOS: export GOOGLE_API_KEY="AIza..."
# ------------------------------------
FORGE_DIR= "gene_forge"
MIND_STATE_DIR = "mind_state"
SANDBOX_DIR= "sandbox"
CUSTOM_GENES_DIR = "custom_genes"
# --- Definición de rutas de archivos ---
PATHS = {
    "kg": os.path.join(MIND_STATE_DIR, "knowledge_graph.pkl"),
    "patterns": os.path.join(MIND_STATE_DIR, "entity_patterns.pkl"),
    "profile": os.path.join(MIND_STATE_DIR, "fitness_profile.pkl"),
    "archive": os.path.join(MIND_STATE_DIR, "strategies_archive.pkl"),
    "dojo_dataset": "dojo_dataset.jsonl",
    "specialists_archive": os.path.join(MIND_STATE_DIR, "specialists_archive.pkl"),
    "general_archive": os.path.join(MIND_STATE_DIR, "general_archive.pkl"), 
    "fitness_profiles": os.path.join(MIND_STATE_DIR, "fitness_profiles.pkl"),
    "self_model": os.path.join(MIND_STATE_DIR, "self_model.pkl"),
    "gene_usage": os.path.join(MIND_STATE_DIR, "gene_usage.pkl"),
    "episodic_memory": os.path.join(MIND_STATE_DIR, "episodic_memory.log"), # <<< AÑADE ESTA LÍNEA
    "genesis_corpus": os.path.join(FORGE_DIR, "genesis_corpus.jsonl"),
}

# Parámetros del Dojo
BATCH_SIZE = 1 # Usamos un tamaño pequeño para ver los ciclos rápidamente
EVOLUTION_FOCUS_SIZE = 1 # Evolucionará sobre el peor de cada lote



import gymnasium as gym
import numpy as np

# (Añadir al principio de las definiciones de clases)
from dataclasses import dataclass, field
import queue
import uuid
from enum import Enum

class GoalStatus(Enum):
    """Define los posibles estados de un objetivo."""
    INACTIVE = "Inactivo"
    ACTIVE = "Activo"
    PAUSED = "Pausado"
    COMPLETED = "Completado"
    FAILED = "Fallido"

# --- UBICACIÓN: Modifica la clase GeneDescriptor ---

@dataclass
class GeneDescriptor:
    """
    Metadatos que describen la capacidad de un Gen para el StrategicPlanner.
    """
    gene_class: type
    purpose_description: str
    relevant_intents: List[str] = field(default_factory=list)
    input_variables: List[str] = field(default_factory=list)
    output_variable: str | None = None
    is_terminal: bool = False
    priority: int = 10 

@dataclass
class Goal:
    """
    Representa una ambición o meta con un ciclo de vida, jerarquía y prioridad.
    """
    description: str
    target: Any  # El estado del mundo que cumple el objetivo
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = 10
    creation_timestamp: float = field(default_factory=time.time)
    parent_goal_id: str | None = None
    sub_goals: List['Goal'] = field(default_factory=list)

    def __repr__(self):
        return f"<Goal(id={self.id[:4]}..., desc='{self.description}', status={self.status.value})>"

@dataclass
class Percept:
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)

class Gene:
    """Clase base para todos los genes."""
    def execute(self, context: dict):
        raise NotImplementedError

# REEMPLAZA tu clase ExecutionContext con esta versión mejorada
# Puedes añadir esta clase cerca del principio, junto a otras definiciones de clases.

class CodeVisitor(ast.NodeVisitor):
    """
    Recorre el Árbol de Sintaxis Abstracta (AST) de un archivo de código
    para extraer nodos (clases, funciones) y aristas (herencia, llamadas, etc.).
    """
    def __init__(self, module_name):
        self.module_name = module_name
        self.nodes = []
        self.edges = []
        self.current_class = None
        self.current_function = None

    def _add_node(self, node_id, node_type, file_path, props=None):
        base_props = {'type': node_type, 'file': file_path}
        if props:
            base_props.update(props)
        self.nodes.append({'id': node_id, 'props': base_props})

    def _add_edge(self, source, target, rel_type, props=None):
        self.edges.append({
            'source': source,
            'target': target,
            'type': rel_type.upper(),
            'props': props or {}
        })

    def visit_ClassDef(self, node):
        class_name = f"{self.module_name}.{node.name}"
        self._add_node(class_name, 'Clase', self.module_name)

        # Relación de herencia
        for base in node.bases:
            try:
                # ast.unparse es una forma sencilla de obtener el nombre de la clase base
                base_name = ast.unparse(base)
                # Asumimos que la clase base puede estar en otro módulo, no la prefijamos
                self._add_edge(class_name, base_name, 'INHERITS_FROM')
            except Exception:
                pass # Ignorar bases complejas

        # Procesar los métodos y sub-clases dentro de esta clase
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
        # Determinar si es una función o un método
        if self.current_class:
            function_name = f"{self.current_class}.{node.name}"
            owner_name = self.current_class
            rel_type = 'DEFINES_METHOD'
        else:
            function_name = f"{self.module_name}.{node.name}"
            owner_name = self.module_name
            rel_type = 'DEFINES_FUNCTION'
        
        self._add_node(function_name, 'Funcion', self.module_name, {'args': [arg.arg for arg in node.args.args]})
        self._add_edge(owner_name, function_name, rel_type)

        # Procesar las llamadas a funciones dentro de esta función
        self.current_function = function_name
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if not self.current_function:
            return # Solo nos interesan llamadas dentro de funciones/métodos

        try:
            # Intentamos obtener el nombre completo de lo que se está llamando
            callee_name = ast.unparse(node.func)
            
            # Si es una instanciación de clase (empieza con mayúscula)
            func_name_part = callee_name.split('.')[-1]
            if func_name_part[0].isupper():
                self._add_edge(self.current_function, callee_name, 'CREATES_INSTANCE')
            else: # Es una llamada a función/método
                self._add_edge(self.current_function, callee_name, 'CALLS')
        except Exception:
            pass # Ignorar llamadas complejas que no se pueden unparse

    def visit_Import(self, node):
        for alias in node.names:
            self._add_edge(self.module_name, alias.name, 'IMPORTS')

    def visit_ImportFrom(self, node):
        module = node.module or 'built-in'
        self._add_edge(self.module_name, module, 'IMPORTS_FROM')


class ExecutionContext:
    def __init__(self, initial_vars: Dict[str, Any] = None):
        # Nos aseguramos de que 'memory' siempre sea un diccionario
        if initial_vars is not None and isinstance(initial_vars, dict):
            self.memory: Dict[str, Any] = initial_vars
        else:
            self.memory: Dict[str, Any] = {}
            
        self.final_answer_text: str = ""
        self.thought_log: List[str] = []

    def set_final_response(self, text: str):
        print(f"\n[DEBUG_CONTEXT] INTENTO DE SETEAR RESPUESTA FINAL A: '{str(text)[:150]}...'\n")
        self.final_answer_text = text

    def set(self, key: str, value: Any):
        # --- AÑADIMOS UNA COMPROBACIÓN DE SEGURIDAD ---
        # Esta comprobación previene el error 'str' object does not support item assignment'
        if not isinstance(self.memory, dict):
            print(f"[ERROR_CONTEXT] Abortando 'set': Se intentó modificar el contexto, pero 'self.memory' está corrupto. Tipo actual: {type(self.memory)}")
            # Opcional: podemos registrar el error en el propio contexto si aún es posible
            if isinstance(self.thought_log, list):
                 self.log_thought("FATAL_ERROR", f"Context memory corrupted. Type became {type(self.memory)}")
            return # Detenemos la ejecución de este método para evitar el crash
            
        self.memory[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if not isinstance(self.memory, dict):
            print(f"[ERROR_CONTEXT] Abortando 'get': Se intentó leer el contexto, pero 'self.memory' está corrupto. Tipo actual: {type(self.memory)}")
            return default
            
        return self.memory.get(key, default)
   
    def get_final_response(self) -> str:
        return self.final_answer_text
    
    def log_thought(self, component: str, reasoning: str):
        self.thought_log.append(f"[{component}]: {reasoning}")

class PerceiveEnvironmentGene(Gene):
    """
    Procesa la observación cruda del entorno y la descompone en conceptos
    comprensibles para la IA dentro del ExecutionContext.
    """
    def __init__(self, observation_var: str = "observation"):
        self.observation_var = observation_var

    def execute(self, context: ExecutionContext):
        observation = context.get(self.observation_var)
        if observation and isinstance(observation, dict):
            agent_pos = observation.get("agent")
            goal_pos = observation.get("goal")
            if agent_pos is not None and goal_pos is not None:
                context.set("agent_pos", tuple(agent_pos))
                context.set("goal_pos", tuple(goal_pos))
                context.set("perceived_world", f"Estoy en {tuple(agent_pos)} y la meta está en {tuple(goal_pos)}.")

# (Al inicio del script)
class FitnessProfile:
    """
    Representa los criterios de juicio para una intención específica.
    Este objeto es evolutivo.
    """
    def __init__(self, intent_name: str):
        self.intent_name = intent_name
        self.weights = {
            # Motivaciones Extrínsecas (basadas en la tarea)
            "similarity": 1.0,
            "efficiency": -10.0,
            "detail": 0.01,
            # Motivaciones Intrínsecas (auto-generadas)
            "novelty": 0.1,      # Novedad respecto a otras soluciones
            "curiosity": 0.0,    # Recompensa por usar genes poco comunes
            "empowerment": 0.0,   # Recompensa por crear nuevas capacidades
        }
        # Los perfiles para intenciones exploratorias tendrán pesos diferentes
        if self.intent_name in ["EXPLORAR", "AUTO_MEJORA"]:
            self.weights["similarity"] = 0.0 # No hay respuesta ideal
            self.weights["curiosity"] = 50.0  # Recompensa alta por ser curioso
            self.weights["empowerment"] = 1000.0 # Recompensa máxima por crear un gen
            self.weights["efficiency"] = -25.0 # <<< Penalización más fuerte por complejidad
            self.weights["novelty"] = 5.0




    def calculate_total_fitness(self, scores: Dict[str, float]) -> float:
        """Calcula el fitness total aplicando los pesos a las puntuaciones brutas."""
        total_fitness = 0
        for key, weight in self.weights.items():
            total_fitness += scores.get(key, 0.0) * weight
        return total_fitness

    def mutate(self, learning_rate: float = 0.1):
        """Muta aleatoriamente uno de los pesos para evolucionar el criterio de juicio."""
        param_to_mutate = random.choice(list(self.weights.keys()))
        
        # El cambio es proporcional al learning_rate
        change_factor = random.uniform(-learning_rate, learning_rate)
        
        # No dejamos que los pesos positivos se vuelvan negativos y viceversa
        if self.weights[param_to_mutate] >= 0:
            self.weights[param_to_mutate] = max(0, self.weights[param_to_mutate] * (1 + change_factor))
        else: # Para pesos negativos como la eficiencia
            self.weights[param_to_mutate] *= (1 + change_factor)
            
        print(f"  [MUTATE_PROFILE] Perfil '{self.intent_name}' mutado. Nuevo peso para '{param_to_mutate}': {self.weights[param_to_mutate]:.2f}")

    def __repr__(self):
        return f"<FitnessProfile({self.intent_name}) | W_sim={self.weights['similarity']:.1f}, W_nov={self.weights['novelty']:.1f}>"
# #############################################################################
# PARTE 1: ARQUITECTURA CENTRAL
# #############################################################################
class MetacognitionEngine:
    """
    Analiza el rendimiento y evoluciona los perfiles de fitness.
    """
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind

    def analyze_and_evolve_profiles(self, batch_results: List[Dict]):
        """
        Revisa los resultados de un lote y muta los perfiles de fitness correspondientes.
        """
        print("\n===== CICLO DE METACOGNICIÓN: EVOLUCIONANDO CRITERIOS DE JUICIO =====")
        
        # Agrupar resultados por intención
        results_by_intent = {}
        for res in batch_results:
            intent = res.get('intent')
            if intent:
                results_by_intent.setdefault(intent, []).append(res['score'])

        for intent, scores in results_by_intent.items():
            if not scores: continue
            
            avg_score = sum(scores) / len(scores)
            profile = self.mind.get_profile_for_intent(intent)
            
            # Meta-fitness: Si el rendimiento promedio es bajo, la tasa de aprendizaje (mutación) es alta.
            # Normalizamos el score (0-1000) a una tasa de aprendizaje (ej. 0.05 - 0.5)
            # Si el score es 1000 (perfecto), learning_rate es bajo. Si es 0, learning_rate es alto.
            learning_rate = 0.5 * (1 - (avg_score / 1000.0))
            
            print(f"  Intención '{intent}': Rendimiento promedio={avg_score:.2f}. Tasa de mutación del perfil={learning_rate:.3f}")
            profile.mutate(learning_rate)
        
        print("====================== METACOGNICIÓN FINALIZADA ======================\n")
# (Asegurarse de que 'from collections import Counter' está al principio del script)

class KnowledgeGraph:
    """
    Implementación del KnowledgeGraph respaldada por una base de datos Neo4j.
    AHORA ES COMPATIBLE CON MULTIPROCESSING.
    """
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="prometheus_password"):
        print("[MIND] Conectando al Knowledge Graph en Neo4j...")
        # Guardamos los detalles de la conexión para la reserialización
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None # Se inicializa en _connect
        self._connect()


    
    def create_vector_index(self):
        """
        Crea un índice vectorial en Neo4j para búsquedas de similitud semántica.
        """
        print("[KG] Creando índice vectorial si no existe...")
        # Nota: Las dimensiones del vector (384) deben coincidir con el modelo que usas
        # ('paraphrase-multilingual-MiniLM-L12-v2' tiene 384 dimensiones).
        index_query = """
        CREATE VECTOR INDEX `concept_embeddings` IF NOT EXISTS
        FOR (c:Concept) ON (c.embedding)
        OPTIONS { indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }}
        """
        try:
            with self.driver.session(database="neo4j") as session:
                session.run(index_query)
            print("[KG] Índice vectorial 'concept_embeddings' asegurado.")
        except Exception as e:
            print(f"[WARN] No se pudo crear el índice vectorial. "
                  f"Asegúrate de que tu versión de Neo4j (5.11+) y GDS son compatibles. Error: {e}")

    def find_similar_nodes(self, vector: List[float], top_k: int = 5) -> List[dict]:
        """
        Encuentra los 'top_k' nodos más similares a un vector dado usando el índice.
        """
        query = """
        CALL db.index.vector.queryNodes('concept_embeddings', $top_k, $vector)
        YIELD node, score
        RETURN node.id AS id, node.definition AS definition, score
        """
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, top_k=top_k, vector=vector)
                return [record.data() for record in result]
        except Exception as e:
            print(f"[ERROR KG] Falló la búsqueda vectorial: {e}")
            # Si el índice no existe o falla, devolvemos una lista vacía para no bloquear el programa.
            return []
    
    
    # EN: la clase KnowledgeGraph
# REEMPLAZA el método get_outgoing_relations con este:

    # EN: la clase KnowledgeGraph
# REEMPLAZA el método get_outgoing_relations con este:

    def get_outgoing_relations(self, node_id: str) -> List[Dict]:
        """
        Recupera todas las relaciones salientes de un nodo, incluyendo el
        tipo de relación, sus propiedades y el nodo destino.
        """
        node_id_lower = node_id.lower().strip()
        query = """
            MATCH (n:Concept {id: $node_id})-[r]->(m:Concept)
            RETURN type(r) as rel_type, properties(r) as properties, m.id as target_node_id
            """
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, node_id=node_id_lower)
                # La siguiente línea es CRUCIAL. record.data() convierte cada
                # resultado en un diccionario, que es lo que el resto del código espera.
                # Aseguramos que sea una lista de diccionarios.
                relations_list = [record.data() for record in result]
                # print(f"DEBUG KG: Relaciones para '{node_id_lower}': {relations_list[:1]}") # Debugging
                return relations_list
        except Exception as e:
            print(f"[ERROR KG] No se pudieron obtener las relaciones para '{node_id_lower}': {e}")
            return []

    def add_batch(self, nodes: List[Dict], edges: List[Dict]):
        """
        Añade nodos y aristas a Neo4j en un lote masivo usando UNWIND.
        Esta versión es robusta contra tipos de relación inválidos.
        """
        with self.driver.session(database="neo4j") as session:
            # Añadir nodos en un lote
            if nodes:
                session.run("""
                    UNWIND $nodes AS node_data
                    MERGE (c:Concept {id: node_data.id})
                    ON CREATE SET c += node_data.props
                    ON MATCH SET c += node_data.props
                """, nodes=nodes)

            # Añadir aristas en un lote
            if edges:
                for edge in edges:
                    # 1. Limpia caracteres inválidos (esto ya lo tenías)
                    rel_type_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', edge['type'])

                    # 2. Omite si la relación está vacía o es muy corta
                    if not rel_type_cleaned or len(rel_type_cleaned) < 2:
                        continue

                    # 3. *** NUEVA LÍNEA DE CÓDIGO ***
                    # Si después de limpiar, empieza con un número, le añadimos un prefijo.
                    if rel_type_cleaned[0].isdigit():
                        rel_type_cleaned = "rel_" + rel_type_cleaned

                    # 4. Construir y ejecutar la consulta Cypher
                    query = f"""
                        MATCH (source:Concept {{id: $source_id}})
                        MATCH (target:Concept {{id: $target_id}})
                        MERGE (source)-[r_new:{rel_type_cleaned}]->(target)
                        SET r_new += $props
                    """
                    session.run(query,
                                source_id=edge['source'],
                                target_id=edge['target'],
                                props=edge.get('props', {}))
    def _connect(self):
        """Inicializa o restablece la conexión con la base de datos."""
        try:
            # CORRECCIÓN: Comprueba si el atributo existe antes de intentar usarlo.
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
            
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            self.driver.verify_connectivity()
            print(f"[MIND pid={os.getpid()}] Conexión con Neo4j establecida con éxito.")
            self._create_constraints()
        except Exception as e:
            print(f"[ERROR FATAL pid={os.getpid()}] No se pudo conectar a la base de datos Neo4j: {e}")
            raise

    def __getstate__(self):
        """
        Define cómo se 'picklea' el objeto. Excluimos el objeto 'driver' no serializable.
        """
        state = self.__dict__.copy()
        del state['driver']
        return state

    def __setstate__(self, state):
        """
        Define cómo se 'unpicklea' el objeto. Recreamos la conexión.
        """
        self.__dict__.update(state)
        self._connect() # Restablece la conexión en el nuevo proceso.

    def _create_constraints(self):
        """Asegura que los nodos de conceptos sean únicos para mejorar el rendimiento."""
        with self.driver.session(database="neo4j") as session:
            session.run("CREATE CONSTRAINT concept_uniqueness IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")

    def close(self):
        """Cierra la conexión con la base de datos."""
        if self.driver:
            self.driver.close()
            print("[MIND] Conexión con Neo4j cerrada.")

    # ... (El resto de los métodos de KnowledgeGraph: add_node_with_definition, add_edge, etc., permanecen exactamente iguales) ...

    def add_node_with_definition(self, node_id: str, definition: str):
        if definition and len(definition) > 10: 
            self.add_node_if_not_exists(node_id, definition=definition)
        else:
            self.add_node_if_not_exists(node_id)

    def add_node_if_not_exists(self, node_id: str, **attrs):
        node_id_lower = node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            session.run("""
                MERGE (c:Concept {id: $id})
                ON CREATE SET c += $props
                ON MATCH SET c += $props
                """, id=node_id_lower, props=attrs)

    def add_edge(self, source_id: str, target_id: str, relationship_type: str, **attrs):
        source_id_lower = source_id.lower().strip()
        target_id_lower = target_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            query = f"""
                MATCH (source:Concept {{id: $source_id}})
                MATCH (target:Concept {{id: $target_id}})
                MERGE (source)-[r:{relationship_type}]->(target)
                SET r += $props
            """
            session.run(query, source_id=source_id_lower, target_id=target_id_lower, props=attrs)

    def get_node_definition(self, node_id: str) -> str | None:
        node_id_lower = node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            result = session.run("MATCH (c:Concept {id: $id}) RETURN c.definition AS definition", id=node_id_lower)
            record = result.single()
            return record["definition"] if record else None

    def find_shortest_path(self, start_node_id: str, end_node_id: str) -> list[str] | None:
        start_id = start_node_id.lower().strip()
        end_id = end_node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (start:Concept {id: $start_id}), (end:Concept {id: $end_id})
                CALL apoc.path.shortestPath(start, end, '>', 5) YIELD path
                RETURN [node IN nodes(path) | node.id] AS path_nodes
                """, start_id=start_id, end_id=end_id)
            record = result.single()
            return record["path_nodes"] if record else None

    def has_node(self, node_id: str) -> bool:
        node_id_lower = node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            result = session.run("MATCH (c:Concept {id: $id}) RETURN count(c) > 0 AS exists", id=node_id_lower)
            return result.single()["exists"]

    def get_all_neighbors(self, node_id: str) -> list[str]:
        node_id_lower = node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            result = session.run("""
                MATCH (c:Concept {id: $id})-[_]-(neighbor:Concept)
                RETURN COLLECT(DISTINCT neighbor.id) AS neighbors
                """, id=node_id_lower)
            record = result.single()
            return record["neighbors"] if record else []
        
class Chromosome:
    def __init__(self, genes: List[Gene], description: str):
        self.genes, self.description = genes, description
        self.fitness: float = 0.0
        self.final_context: ExecutionContext = None
    def __repr__(self):
        return f"<Chromosome ('{self.description}') | Fitness: {self.fitness:.2f}>"
class PatternFinderGene(Gene):
    """
    Analiza un archivo de cromosomas para encontrar patrones recurrentes (secuencias de genes).
    Realiza una forma de introspección sobre las estrategias exitosas de la IA.
    """
    def __init__(self, archive_to_analyze: str, intent_var: str, output_var: str, mind: 'PrometheusAGI', top_n: int = 3):
        self.archive_to_analyze = archive_to_analyze
        self.intent_var = intent_var
        self.output_var = output_var
        self.mind = mind
        self.top_n = top_n

    def execute(self, context: ExecutionContext):
        intent_to_analyze = context.get(self.intent_var)
        
        target_chromosomes = []
        if self.archive_to_analyze == "specialists":
            if not intent_to_analyze:
                context.set(self.output_var, "Se necesita especificar una intención para analizar a los especialistas.")
                return
            target_chromosomes = self.mind.get_specialists_for_intent(intent_to_analyze)
        elif self.archive_to_analyze == "general":
            target_chromosomes = self.mind.successful_strategies_archive
        
        if not target_chromosomes:
            context.set(self.output_var, f"No encontré estrategias en el archivo '{self.archive_to_analyze}' para analizar.")
            return

        # Extraer secuencias de nombres de genes
        gene_sequences = [[gene.__class__.__name__ for gene in ch.genes] for ch in target_chromosomes]
        
        # Minería de patrones (n-gramas)
        all_bigrams = []
        all_trigrams = []
        for seq in gene_sequences:
            if len(seq) >= 2:
                all_bigrams.extend(zip(seq, seq[1:]))
            if len(seq) >= 3:
                all_trigrams.extend(zip(seq, seq[1:], seq[2:]))

        bigram_counts = Counter(all_bigrams).most_common(self.top_n)
        trigram_counts = Counter(all_trigrams).most_common(self.top_n)

        # Formatear la salida para que sea legible
        report_parts = ["He analizado mis estrategias exitosas y he descubierto los siguientes patrones recurrentes:"]
        if bigram_counts:
            report_parts.append("\n--- Bigramas Más Comunes (Secuencias de 2 Pasos) ---")
            for (g1, g2), count in bigram_counts:
                report_parts.append(f"  - La secuencia '{g1} -> {g2}' ha aparecido {count} veces.")
        
        if trigram_counts:
            report_parts.append("\n--- Trigramas Más Comunes (Secuencias de 3 Pasos) ---")
            for (g1, g2, g3), count in trigram_counts:
                report_parts.append(f"  - La secuencia '{g1} -> {g2} -> {g3}' ha aparecido {count} veces.")

        if not bigram_counts and not trigram_counts:
            report_parts.append("No encontré patrones suficientemente recurrentes.")
            
        final_report = "\n".join(report_parts)
        context.set(self.output_var, final_report)
        print(f"  [Gene] PatternFinder: Análisis completado. {len(bigram_counts) + len(trigram_counts)} patrones principales encontrados.")


class SyntaxVisitor(ast.NodeVisitor):
    """
    Un visitante de nodos AST especializado en extraer información semántica
    de alto nivel sobre el código de un Gen.
    """
    def __init__(self):
        self.analysis = {
            "imports": set(),
            "function_calls": Counter(),
            "loops": 0,
            "conditionals": 0,
            "context_gets": set(),
            "context_sets": set()
        }

    def _get_full_call_name(self, node: ast.Call) -> str:
        """Intenta reconstruir el nombre completo de una llamada, ej: 'requests.get'"""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            # Reconstrucción recursiva para llamadas anidadas como 'self.mind.goal_manager.add_goal'
            try:
                return f"{self._get_full_call_name(node.func.value)}.{node.func.attr}"
            except:
                 # En casos muy complejos, nos rendimos de forma controlada
                return f"Complex.{node.func.attr}"
        return "UnknownFunctionCall"

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.analysis["imports"].add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        self.analysis["imports"].add(node.module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        full_name = self._get_full_call_name(node)
        self.analysis["function_calls"][full_name] += 1
        
        # Específicamente buscamos interacciones con el ExecutionContext
        if full_name.endswith("context.get") and node.args:
            if isinstance(node.args[0], ast.Constant):
                self.analysis["context_gets"].add(node.args[0].value)
        
        if full_name.endswith("context.set") and node.args:
            if isinstance(node.args[0], ast.Constant):
                self.analysis["context_sets"].add(node.args[0].value)

        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.analysis["loops"] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self.analysis["conditionals"] += 1
        self.generic_visit(node)

# Reemplaza la clase RecallLastResponseGene con esta versión

class RecallLastResponseGene(Gene):
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

class AnalyzeGeneSyntaxGene(Gene):
    """
    Un gen de introspección que analiza el código fuente de otro gen usando AST
    para entender su estructura y comportamiento.
    """
    def __init__(self, gene_name_var: str, output_var: str, gene_map: dict[str, type]):
        self.gene_name_var = gene_name_var
        self.output_var = output_var
        self.gene_map = gene_map

    def _format_report(self, gene_name: str, analysis: dict) -> str:
        """Convierte los datos del análisis en un informe legible para el usuario."""
        lines = [f"He realizado un análisis sintáctico de mi gen '{gene_name}' y esto es lo que he encontrado:"]
        lines.append("-" * 60)
        
        if analysis["imports"]:
            lines.append(f"📚 Dependencias Externas (Módulos): {', '.join(sorted(list(analysis['imports'])))}")
        
        lines.append(f"⚙️ Lógica Interna: Contiene {analysis['conditionals']} estructura(s) condicional(es) (if) y {analysis['loops']} bucle(s).")

        if analysis["context_gets"]:
            lines.append(f"📥 Variables de Entrada (del contexto): {', '.join(sorted(list(analysis['context_gets'])))}")
        
        if analysis["context_sets"]:
            lines.append(f"📤 Variables de Salida (al contexto): {', '.join(sorted(list(analysis['context_sets'])))}")

        if analysis["function_calls"]:
            lines.append("\n📞 Llamadas a Funciones Clave (las 5 más comunes):")
            for func, count in analysis["function_calls"].most_common(5):
                lines.append(f"  - Llama a '{func}' ({count} ve(z|ces))")

        lines.append("-" * 60)
        lines.append("Este análisis me ayuda a entender cómo funcionan mis propias capacidades.")
        return "\n".join(lines)

    def execute(self, context: ExecutionContext):
        gene_to_analyze_raw = context.get("main_topic")
        
        if not gene_to_analyze_raw:
            context.set(self.output_var, "No se especificó qué gen analizar.")
            context.set_final_response("No se especificó qué gen analizar.")
            return

        gene_to_analyze_lower = gene_to_analyze_raw.lower()
        
        # Creamos un mapa temporal con todas las claves en minúsculas para una comparación robusta.
        lower_case_gene_map = {k.lower(): v for k, v in self.gene_map.items()}

        target_class = lower_case_gene_map.get(gene_to_analyze_lower)

        if not target_class:
            error_msg = f"Lo siento, no tengo un gen llamado '{gene_to_analyze_raw}' en mi genoma."
            context.set(self.output_var, error_msg)
            context.set_final_response(error_msg)
            return
            
        try:
            source_code = inspect.getsource(target_class)
            tree = ast.parse(source_code)
            visitor = SyntaxVisitor()
            visitor.visit(tree)
            
            # Usamos el nombre correcto de la clase (con mayúsculas) para el informe.
            report_string = self._format_report(target_class.__name__, visitor.analysis)
            context.set(self.output_var, report_string)
            context.set_final_response(report_string)

        except Exception as e:
            error_message = f"Encontré un error al intentar analizar mi propio código para el gen '{target_class.__name__}': {e}"
            context.set(self.output_var, error_message)
            context.set_final_response(error_message)

class WhyDidIFailGene(Gene):
    def __init__(self, failure_analyzer: 'FailureAnalysisEngine', output_var: str = "failure_report"):
        self.failure_analyzer = failure_analyzer
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        diagnosis = self.failure_analyzer.analyze_last_failure()
        context.set(self.output_var, diagnosis)

class CompositeGene(Gene):
    def __init__(self, name: str, sequence: List[Gene]):
        self.name, self.sequence = name, sequence
    def execute(self, context: ExecutionContext):
        for gene in self.sequence:
            gene.execute(context)
            if context.get("execution_error"): break
    def __repr__(self): return f"<CompositeGene({self.name})>"

class StyleAnalyzerGene(Gene):
    def execute(self, context: ExecutionContext):
        text = context.get("query", "").lower()
        formality = "informal" if any(w in text for w in ["tío", "osea", "que tal"]) else "formal"
        context.set("user_style_profile", {"formality": formality})

class CuriosityGene(Gene):
    """
    Establece un objetivo interno en el GoalManager basado en la exploración.
    """
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind
        self.goal_manager = mind.goal_manager

    def execute(self, context: ExecutionContext):
        # Si ya hay un objetivo, no generamos uno nuevo pero sí informamos de ello.
        if active_goal := self.goal_manager.get_active_goal():
            context.set("self_generated_goal", f"Ya tengo un objetivo activo: {active_goal.description}")
            return

        # 1. Elige un objetivo (igual que antes)
        corner = random.choice([[0,0], [0, self.mind.env_size-1], [self.mind.env_size-1, 0], [self.mind.env_size-1, self.mind.env_size-1]])
        target_pos = np.array(corner)
        description = f"Explorar la esquina en {target_pos}"

        # 2. Llama a add_goal y captura el objeto que devuelve
        new_goal = self.goal_manager.add_goal(
            description=description,
            target=target_pos
        )
        
        # 3. Usa el objeto devuelto para actualizar el contexto
        context.set("self_generated_goal", new_goal.description)

# REEMPLAZA la clase CognitiveCoreGene completa con esta versión:

class CognitiveCoreGene(Gene):
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind
        self.kg = mind.knowledge_graph
        self.similarity_model = mind.similarity_model
        self.nlp_processor = None  # Se inicializará de forma perezosa
        self.synthesizer = GraphTraversalSynthesizer(mind)
    
    @property
    def nlp(self):
        """Inicialización perezosa del procesador NLP."""
        if self.nlp_processor is None:
            self.nlp_processor = self.mind.intent_engine.nlp
        return self.nlp_processor

    # EN: la clase CognitiveCoreGene
# REEMPLAZA el método _analyze_query con este:

    def _analyze_query(self, query: str) -> Tuple[List[float], List[str]]:
        """Convierte la consulta en un vector y extrae entidades y keywords limpias."""
        query_vector = self.similarity_model.encode(query).tolist()
        doc = self.nlp(query)  # Usar la propiedad nlp en lugar de nlp_processor
        
        # --- LÓGICA DE FILTRADO MEJORADA ---
        keywords = []
        stop_words = {'es', 'son', 'que', 'un', 'una', 'el', 'la', 'los', 'las', 'de', 'del', 'a', 'al'}
        
        # Añadir entidades nombradas
        keywords.extend([ent.text.lower() for ent in doc.ents])
        
        # Añadir trozos de sustantivos, pero solo si no son palabras vacías
        for chunk in doc.noun_chunks:
            # Filtramos palabras vacías y muy cortas
            if chunk.text.lower() not in stop_words and len(chunk.text) > 2:
                keywords.append(chunk.text.lower())
        
        # Añadir tokens importantes que no sean stop words
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                token.text.lower() not in stop_words and 
                len(token.text) > 2 and
                not token.is_punct):
                keywords.append(token.text.lower())
        # --- FIN DE LA LÓGICA MEJORADA ---

        return query_vector, list(set(keywords))

    def _find_start_nodes(self, vector: List[float], keywords: List[str]) -> List[Dict]:
        """Encuentra nodos de inicio combinando búsqueda vectorial y por keywords."""
        similar_nodes = self.kg.find_similar_nodes(vector, top_k=3)
        found_nodes = {node['id']: node for node in similar_nodes}
        
        # Buscar por keywords exactos y similares
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            
            # Búsqueda exacta
            if keyword_lower not in found_nodes and self.kg.has_node(keyword_lower):
                found_nodes[keyword_lower] = {'id': keyword_lower, 'score': 0.99}
            
            # Búsqueda parcial para términos como "mamifero" vs "mamífero"
            if keyword_lower not in found_nodes:
                # Buscar variaciones comunes
                variations = [
                    keyword_lower.replace('í', 'i').replace('é', 'e').replace('á', 'a').replace('ó', 'o').replace('ú', 'u'),
                    keyword_lower.replace('i', 'í').replace('e', 'é').replace('a', 'á').replace('o', 'ó').replace('u', 'ú')
                ]
                
                for variation in variations:
                    if variation != keyword_lower and self.kg.has_node(variation):
                        found_nodes[variation] = {'id': variation, 'score': 0.95}
                        break
        
        # Si no encontramos nodos, intentar búsqueda más amplia
        if not found_nodes:
            print(f"[CognitiveCore] No se encontraron nodos para keywords: {keywords}")
            print("[CognitiveCore] Intentando búsqueda más amplia...")
            
            # Buscar nodos que contengan las keywords
            broader_search = self._search_nodes_containing_keywords(keywords)
            for node in broader_search:
                found_nodes[node['id']] = node
        
        result = sorted(found_nodes.values(), key=lambda x: x.get('score', 0), reverse=True)
        print(f"[CognitiveCore] Nodos encontrados: {[n['id'] for n in result]}")
        return result
        
    def _search_nodes_containing_keywords(self, keywords: List[str]) -> List[Dict]:
        """Búsqueda más amplia que encuentra nodos que contengan las keywords."""
        results = []
        for keyword in keywords:
            keyword_lower = keyword.lower().strip()
            try:
                with self.kg.driver.session(database="neo4j") as session:
                    # Buscar en id, name y description
                    query = """
                    MATCH (c:Concept)
                    WHERE toLower(c.id) CONTAINS $keyword 
                       OR toLower(c.name) CONTAINS $keyword 
                       OR toLower(c.description) CONTAINS $keyword
                    RETURN c.id AS id, c.name AS name, c.description AS definition, 0.8 AS score
                    LIMIT 3
                    """
                    result = session.run(query, keyword=keyword_lower)
                    for record in result:
                        results.append(record.data())
            except Exception as e:
                print(f"[ERROR] Fallo en búsqueda amplia para '{keyword}': {e}")
        
        return results

    def _needs_tool(self, response: str) -> bool:
        if not response or len(response) < 60:
            print("[CognitiveCore] Autoevaluación: La respuesta interna es insuficiente. Se necesita una herramienta.")
            return True
        return False

    def _select_best_tool(self, query: str) -> Gene | None:
        print("[CognitiveCore] Buscando la mejor herramienta para la consulta...")
        query_embedding = self.similarity_model.encode(query)
        best_tool_name, highest_score = None, -1.0
        for descriptor in self.mind.tool_descriptors:
            score = util.pytorch_cos_sim(query_embedding, self.similarity_model.encode(descriptor.purpose_description)).item()
            print(f"  - Herramienta candidata: {descriptor.gene_class.__name__} (Relevancia: {score:.2f})")
            if score > highest_score:
                highest_score, best_tool_name = score, descriptor.gene_class.__name__
        if highest_score > 0.4:
            print(f"[CognitiveCore] Herramienta seleccionada: {best_tool_name} con score {highest_score:.2f}")
            return self.mind.tool_genes.get(best_tool_name)
        print("[CognitiveCore] Ninguna herramienta parece suficientemente relevante.")
        return None

    def execute(self, context: ExecutionContext):
        query = context.get("query")
        if not query: return

        print("[CognitiveCore] Iniciando ciclo de pensamiento autónomo...")
        final_answer = None
        
        # Paso 1: Búsqueda y síntesis interna
        query_vector, keywords = self._analyze_query(query)
        start_nodes = self._find_start_nodes(query_vector, keywords)
        print(f"[CognitiveCore] Nodos iniciales encontrados en KG: {[n['id'] for n in start_nodes]}")
        internal_answer = self.synthesizer.synthesize_answer(query, start_nodes)
        print(f"[CognitiveCore] Respuesta interna generada: '{internal_answer[:100]}...'")
        
        # Paso 2: Autoevaluación y uso de herramientas
        if self._needs_tool(internal_answer):
            tool_gene = self._select_best_tool(query)
            if tool_gene:
                print(f"[CognitiveCore] Ejecutando herramienta '{tool_gene.__class__.__name__}'...")
                tool_gene.execute(context)
                
                web_summary = context.get('web_summary')
                if web_summary:
                    final_answer = f"Según mi investigación en la web: {web_summary}"
                else:
                    final_answer = "Busqué información con mis herramientas, pero no pude obtener un resultado claro. " + internal_answer
            else:
                final_answer = internal_answer
        else:
            final_answer = internal_answer

        context.set_final_response(final_answer)
class GraphTraversalSynthesizer:
    """
    Genera texto coherente realizando un "paseo" guiado por el Knowledge Graph,
    utilizando un modelo de similitud para elegir el camino más relevante.
    """
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind
        self.kg = mind.knowledge_graph
        self.similarity_model = mind.similarity_model

    def _get_most_relevant_path(self, start_node_id: str, query_embedding: np.ndarray) -> List[Dict] | None:
        """
        Encuentra la ruta saliente más relevante desde un nodo inicial,
        comparando la consulta del usuario con el texto de las relaciones.
        """
        relations = self.kg.get_outgoing_relations(start_node_id)
        if not relations:
            return None

        best_path = None
        highest_score = -1.0

        for rel in relations:
            # --- INICIO DE LA CORRECCIÓN CLAVE ---
            # Esta línea DEBE estar aquí, justo antes de intentar usar .get() en 'rel'.
            # Si 'rel' no es un diccionario, se saltará al siguiente elemento.
            if not isinstance(rel, dict):
                print(f"[ERROR KG] Elemento de relación inesperado en _get_most_relevant_path: {type(rel)} - {rel}. Se esperaba un diccionario. Saltando este elemento.")
                continue # Saltar este elemento malformado
            # --- FIN DE LA CORRECCIÓN CLAVE ---

            relation_text = rel.get('properties', {}).get('text', rel.get('rel_type', ''))
            if not relation_text:
                continue

            relation_embedding = self.similarity_model.encode(relation_text)
            score = util.pytorch_cos_sim(query_embedding, relation_embedding).item()

            if score > highest_score:
                highest_score = score
                best_path = [
                    {'type': 'node', 'value': start_node_id},
                    {'type': 'relation', 'value': relation_text},
                    {'type': 'node', 'value': rel.get('target_node_id')}
                ]
        
        return best_path if highest_score > 0.4 else None # Umbral de relevancia

    def synthesize_answer(self, query: str, start_nodes: List[Dict]) -> str:
        """
        Genera una o más frases a partir de los nodos iniciales más relevantes.
        """
        if not start_nodes:
            return "No encontré un punto de partida en mi conocimiento para formular una respuesta."

        query_embedding = self.similarity_model.encode(query)
        final_sentences = []

        # Intentamos formular una respuesta para cada uno de los nodos iniciales
        for start_node in start_nodes[:2]: # Limitamos a los 2 nodos más relevantes
            start_node_id = start_node['id']
            
            # Buscamos el camino más lógico desde este nodo
            best_path = self._get_most_relevant_path(start_node_id, query_embedding)

            if best_path:
                # Ensamblamos la frase a partir del camino encontrado
                # path = [{'type': 'node', 'value': 'einstein'}, {'type': 'relation', 'value': 'desarrolló'}, {'type': 'node', 'value': 'relatividad'}]
                # resultado -> "Einstein desarrolló relatividad."
                sentence_parts = [p['value'] for p in best_path]
                sentence = " ".join(sentence_parts).capitalize() + "."
                final_sentences.append(sentence)
        
        if not final_sentences:
            return f"Tengo información sobre '{start_nodes[0]['id']}', pero no pude formular una respuesta específica a tu pregunta con las relaciones que conozco."

        return " ".join(final_sentences)
class LinguisticAnalysisGene(Gene):
    """
    Analiza un texto para extraer sus propiedades lingüísticas
    y las integra en el KnowledgeGraph. Es compatible con multiprocessing.
    """
    # Atributos de clase para las instancias de fallback en el proceso principal
    _fallback_kg = None
    _fallback_nlp = None

    def __init__(self, graph: KnowledgeGraph, nlp_processor):
        self.graph = graph
        self.nlp = nlp_processor
    def execute(self, context: ExecutionContext):
        # --- Obtención de recursos ---
        try:
            kg = worker_kg
        except NameError:
            if LinguisticAnalysisGene._fallback_kg is None:
                print("[WARN] LinguisticAnalysisGene ejecutándose en proceso principal. Creando instancia de KnowledgeGraph de fallback.")
                LinguisticAnalysisGene._fallback_kg = KnowledgeGraph()
            kg = LinguisticAnalysisGene._fallback_kg

        try:
            nlp = worker_nlp
        except NameError:
            if LinguisticAnalysisGene._fallback_nlp is None:
                print("[WARN] LinguisticAnalysisGene ejecutándose en proceso principal. Creando instancia de spaCy de fallback.")
                LinguisticAnalysisGene._fallback_nlp = spacy.load("es_core_news_sm")
            nlp = LinguisticAnalysisGene._fallback_nlp
        # --- Fin de obtención de recursos ---

        text_to_analyze = context.get("web_summary")
        main_topic = context.get("main_topic")

        if not text_to_analyze or not main_topic:
            return

        print(f"  [Gene] LinguisticAnalysis: Analizando texto sobre '{main_topic}' para enriquecer vocabulario...")
        doc = nlp(text_to_analyze)

        words_processed = 0
        max_words = 50

        for token in doc:
            if token.pos_ in ["NOUN", "ADJ", "VERB"] and not token.is_stop:
                word = token.text.lower()
                lemma = token.lemma_.lower()

                # Usa la instancia 'kg' obtenida
                kg.add_node_if_not_exists(word, type="Palabra", pos=token.pos_)
                kg.add_node_if_not_exists(lemma, type="Concepto")
                kg.add_edge(word, lemma, "es_lema_de")

                if lemma != main_topic:
                    kg.add_edge(main_topic, lemma, "se_relaciona_con")
                
                words_processed += 1
                if words_processed >= max_words:
                    break
        
        print(f"  [Gene] LinguisticAnalysis: Grafo enriquecido con {words_processed} nuevas relaciones lingüísticas.")

class EvolveStrategyGene(Gene):
    """
    Un "meta-gen" que activa un ciclo de evolución interno (un "Dojo") 
    para forjar una estrategia óptima para una consulta compleja en tiempo real.
    """
    def __init__(self, mind: 'PrometheusAGI', generations: int = 10, population_size: int = 20):
        self.mind = mind
        self.generations = generations
        self.population_size = population_size

    def execute(self, context: ExecutionContext):
        query = context.get("query")
        intent = context.get("intent")

        if not query or not intent:
            context.set("evolved_answer", "No puedo evolucionar una estrategia sin una consulta y una intención claras.")
            return

        print(f"\n===== [EvolveStrategyGene] INICIANDO DOJO INTERNO PARA LA CONSULTA: '{query[:50]}...' =====")
        context.log_thought(
            "EvolveStrategyGene", 
            f"La consulta es compleja. Activando ciclo de evolución de {self.generations} generaciones para forjar una respuesta óptima."
        )

        # --- El Reto del Fitness: Crear una "respuesta ideal" para guiar la evolución ---
        # 1. Usamos una búsqueda web para obtener un documento base de alta calidad.
        #    Esto ancla la evolución a la realidad factual.
        web_search_ctx = ExecutionContext(initial_vars={"main_topic": query, "query": query})
        WebSearchGene(query_var="main_topic", output_key="web_summary").execute(web_search_ctx)
        ideal_response_base = web_search_ctx.get("web_summary", "")

        if not ideal_response_base or len(ideal_response_base) < 100:
            print("[EvolveStrategyGene] No se pudo obtener una base de conocimiento sólida de la web. La evolución será puramente exploratoria.")
            ideal_response_base = query # Usamos la propia pregunta como guía mínima.

        # 2. Invocamos el motor de evolución del Dojo.
        best_chromosome = self.dojo._evolve_on_challenge(
            instruction=query,
            ideal_response=ideal_response_base,
            intent=intent,
            generations=self.generations,
            population_size=self.population_size
        )

        # 3. Ejecutamos el mejor cromosoma encontrado y guardamos su respuesta.
        if best_chromosome:
            print("===== [EvolveStrategyGene] DOJO INTERNO COMPLETADO. Ejecutando la estrategia optimizada... =====")
            final_context = ExecutionContext(initial_vars=context.memory)
            
            # Ejecutamos la secuencia de genes del cromosoma ganador
            for gene in best_chromosome.genes:
                gene.execute(final_context)
            
            final_answer = final_context.get_final_response()
            
            # Guardamos la respuesta final en el contexto principal.
            context.set_final_response(final_answer)
            context.log_thought(
                "EvolveStrategyGene", 
                f"Evolución completada. La mejor estrategia encontrada (Fitness: {best_chromosome.fitness:.2f}) ha generado la respuesta final."
            )
        else:
            context.set_final_response("Tras un proceso de reflexión interna, no he podido encontrar una estrategia satisfactoria para tu pregunta.")

class CheckGoalStatusGene(Gene):
    """Consulta el GoalManager y pone el objetivo activo en el contexto."""
    def __init__(self, goal_manager: 'GoalManager', output_var: str = "active_goal"):
        self.goal_manager = goal_manager
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        active_goal = self.goal_manager.get_active_goal()
        if active_goal:
            context.set(self.output_var, active_goal)

# #############################################################################
# PARTE 2: EL ARSENAL DE GENES
# #############################################################################

LEXICON_BRAIN_V2 = {
    "EXPRESS_EMPATHY": [{"text": "Comprendo que la situación pueda ser compleja.", "style": {"formality": "formal"}}, {"text": "Eso suena fatal, de verdad.", "style": {"formality": "informal"}}],
    "OFFER_HELP": [{"text": "¿Hay algo en lo que pueda asistirle?", "style": {"formality": "formal"}}, {"text": "¿Quieres que te dé un ejemplo?", "style": {"formality": "informal"}}],
    "GREETING": [{"text": "Hola. ¿En qué puedo ayudarte?", "style": {"formality": "formal"}}, {"text": "¡Hola! Dime qué necesitas.", "style": {"formality": "informal"}}]
}

# (Añadir esta nueva clase en la sección de Genes del script prometheus_v2.py)
class ScientificSearchGeneAsync(Gene):
    """
    Versión asíncrona de ScientificSearchGene.
    Realiza búsquedas en PubMed de forma no bloqueante.
    """
    def __init__(self, query_var: str = "main_topic", output_key: str = "scientific_summary", max_results: int = 3):
        self.query_var = query_var
        self.output_key = output_key
        self.max_results = max_results
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    async def _search_pubmed(self, session, query: str) -> list[str]:
        search_url = f"{self.base_url}esearch.fcgi"
        params = {'db': 'pubmed', 'term': query, 'retmax': self.max_results, 'retmode': 'json'}
        try:
            async with session.get(search_url, params=params, timeout=10) as response:
                response.raise_for_status()
                data = await response.json()
                return data.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"[WARN] ScientificSearchGeneAsync: Error en búsqueda: {e}")
            return []

    async def _fetch_abstracts(self, session, article_ids: list[str]) -> str:
        if not article_ids: return ""
        fetch_url = f"{self.base_url}efetch.fcgi"
        params = {'db': 'pubmed', 'id': ",".join(article_ids), 'retmode': 'xml', 'rettype': 'abstract'}
        try:
            async with session.get(fetch_url, params=params, timeout=15) as response:
                response.raise_for_status()
                xml_content = await response.text()
                soup = BeautifulSoup(xml_content, 'xml')
                abstracts = soup.find_all('AbstractText')
                return " ".join([abstract.get_text(strip=True) for abstract in abstracts])
        except Exception as e:
            print(f"[WARN] ScientificSearchGeneAsync: Error en fetch: {e}")
            return ""

    async def execute(self, context: ExecutionContext):
        query_text = context.get(self.query_var)
        if not query_text:
            context.set(self.output_key, "No se especificó un tema para la búsqueda científica.")
            return

        print(f"  [Gene] ScientificSearch (Async): Iniciando búsqueda en PubMed sobre '{query_text[:50]}...'")
        
        async with aiohttp.ClientSession() as session:
            article_ids = await self._search_pubmed(session, query_text)
            if not article_ids:
                context.set(self.output_key, f"No encontré artículos en PubMed para '{query_text}'.")
                return

            print(f"  [Gene] ScientificSearch (Async): Se encontraron {len(article_ids)} IDs. Obteniendo resúmenes...")
            combined_abstracts = await self._fetch_abstracts(session, article_ids)
            
            if combined_abstracts:
                context.set(self.output_key, combined_abstracts)
            else:
                context.set(self.output_key, "No se pudieron extraer los resúmenes de los artículos.")



class ScientificSearchGene(Gene):
    """
    Busca artículos en bases de datos científicas (empezando por PubMed)
    y extrae sus resúmenes para el análisis.
    """
    def __init__(self, query_var: str = "main_topic", output_key: str = "scientific_summary", max_results: int = 3):
        self.query_var = query_var
        self.output_key = output_key
        self.max_results = max_results
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def _search_pubmed(self, query: str) -> list[str]:
        """Realiza una búsqueda en PubMed y devuelve los IDs de los artículos."""
        try:
            search_url = f"{self.base_url}esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': query,
                'retmax': self.max_results,
                'retmode': 'json'
            }
            response = requests.get(search_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"[WARN] ScientificSearchGene: Error en la búsqueda de IDs en PubMed: {e}")
            return []

    def _fetch_abstracts(self, article_ids: list[str]) -> str:
        """Recupera los resúmenes (abstracts) de los artículos a partir de sus IDs."""
        if not article_ids:
            return ""
        try:
            fetch_url = f"{self.base_url}efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ",".join(article_ids),
                'retmode': 'xml', # XML es más fácil de parsear para los abstracts
                'rettype': 'abstract'
            }
            response = requests.get(fetch_url, params=params, timeout=15)
            response.raise_for_status()
            
            # Usamos BeautifulSoup para parsear el XML de respuesta
            soup = BeautifulSoup(response.content, 'xml')
            abstracts = soup.find_all('AbstractText')
            
            # Unimos todos los textos de los abstracts encontrados
            full_text = " ".join([abstract.get_text(strip=True) for abstract in abstracts])
            return full_text
            
        except Exception as e:
            print(f"[WARN] ScientificSearchGene: Error al recuperar abstracts de PubMed: {e}")
            return ""

    def execute(self, context: ExecutionContext):
        query_text = context.get(self.query_var)
        if not query_text:
            context.set(self.output_key, "No se especificó un tema para la búsqueda científica.")
            return

        print(f"  [Gene] ScientificSearch: Buscando en PubMed sobre '{query_text[:50]}...'")
        
        # Paso 1: Buscar IDs de artículos
        article_ids = self._search_pubmed(query_text)
        if not article_ids:
            context.set(self.output_key, f"No encontré artículos científicos relevantes en PubMed para '{query_text}'.")
            return
            
        print(f"  [Gene] ScientificSearch: Se encontraron {len(article_ids)} IDs de artículos. Obteniendo resúmenes...")

        # Paso 2: Obtener los resúmenes de esos artículos
        combined_abstracts = self._fetch_abstracts(article_ids)
        
        if combined_abstracts:
            print(f"  [Gene] ScientificSearch: Resúmenes combinados obtenidos ({len(combined_abstracts)} caracteres).")
            context.set(self.output_key, combined_abstracts)
        else:
            context.set(self.output_key, f"Pude encontrar artículos para '{query_text}', pero no pude extraer sus resúmenes.")

# AÑADE esta nueva clase completa al script.

class ResolveContextGene(Gene):
    """
    Analiza la consulta en busca de referencias contextuales como "el tema anterior"
    y utiliza el historial de conversación para resolverlas, actualizando el 'main_topic'.
    """
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind


    def execute(self, context: ExecutionContext):
        query = context.get("query", "").lower()
        
        # Comprobamos si alguna de las frases de activación está en la consulta
        if any(phrase in query for phrase in self.trigger_phrases):
            print("  [Gene] ResolveContext: Se detectó una referencia contextual.")
            
            # Accedemos al historial de conversación de la mente
            # El historial guarda {"user": ..., "prometheus": ...}, nos interesa el penúltimo turno (el último del usuario antes de este)
            if len(self.mind.conversational_history) > 0:
                last_interaction = self.mind.conversational_history[-1]
                # Para ser más precisos, necesitamos el tópico de la última interacción.
                # Re-analizamos la última pregunta del usuario para obtener su tópico.
                last_user_query = last_interaction.get("user")
                if last_user_query:
                    # Usamos el motor de intenciones para obtener el tópico anterior de forma fiable
                    last_analysis = self.mind.intent_engine.analyze(last_user_query)
                    previous_topic = last_analysis.get("main_topic")
                    
                    if previous_topic:
                        print(f"  [Gene] ResolveContext: Tópico resuelto. Reemplazando '{context.get('main_topic')}' con '{previous_topic}'.")
                        # Sobrescribimos el tópico actual con el tópico resuelto del pasado
                        context.set("main_topic", previous_topic)
                    else:
                        print("  [Gene] ResolveContext: No se pudo determinar un tópico claro de la interacción anterior.")
            else:
                print("  [Gene] ResolveContext: No hay historial de conversación para resolver el contexto.")


class ExplainPlanGene(Gene):
    """
    Consulta el objetivo activo en el GoalManager y genera una explicación
    textual de los pasos que se seguirían para alcanzarlo.
    """
    def __init__(self, goal_manager: 'GoalManager'):
        self.goal_manager = goal_manager

    def execute(self, context: ExecutionContext):
        active_goal = self.goal_manager.get_active_goal()
        if not active_goal:
            context.set("plan_explanation", "No tengo un objetivo activo en este momento, por lo que estoy en un estado de espera o exploración libre.")
            return

        # Generamos un plan hipotético basado en el tipo de objetivo
        plan_steps = [f"Mi objetivo actual es: '{active_goal.description}'. Mi plan para lograrlo es el siguiente:"]
        
        if isinstance(active_goal.target, np.ndarray): # Si es un objetivo en el entorno virtual
            plan_steps.append("1. Percibir mi posición actual y la de la meta en el entorno.")
            plan_steps.append(f"2. Calcular la ruta más eficiente desde mi posición hasta la coordenada {active_goal.target.tolist()}.")
            plan_steps.append("3. Ejecutar la secuencia de acciones de movimiento (ARRIBA, ABAJO, etc.) hasta alcanzar la meta.")
            plan_steps.append("4. Al llegar, verificar que mi posición coincide con la del objetivo para marcarlo como completado.")

        elif isinstance(active_goal.target, str): # Si el objetivo es encontrar información
             plan_steps.append(f"1. Analizar el objetivo para extraer el tema clave: '{active_goal.target}'.")
             plan_steps.append("2. Ejecutar una búsqueda web extensiva para recopilar información de múltiples fuentes fiables.")
             plan_steps.append("3. Procesar y resumir la información encontrada para formar una respuesta coherente.")
             plan_steps.append("4. Actualizar mi propia base de conocimiento con los nuevos datos aprendidos.")
        else: # Para otros tipos de objetivos más abstractos
            plan_steps.append("1. Descomponer el objetivo principal en sub-tareas más pequeñas y manejables.")
            plan_steps.append("2. Para cada sub-tarea, buscar los genes y estrategias más adecuados en mi arsenal.")
            plan_steps.append("3. Ejecutar las estrategias en secuencia, verificando el progreso después de cada paso.")

        context.set("plan_explanation", "\n".join(plan_steps))

# EN: la sección de definición de Genes
# REEMPLAZA tu clase DeepReasoningGene actual con esta:

# REEMPLAZA tu clase DeepReasoningGene actual con esta versión completa y corregida:

class DeepReasoningGene(Gene):
    """
    Un gen de razonamiento avanzado que intenta encontrar relaciones complejas
    entre dos conceptos en el Knowledge Graph. Es compatible con multiprocessing.
    """
    _fallback_kg = None
    _fallback_nlp = None

    def __init__(self, graph: KnowledgeGraph, nlp_processor):
        self.graph = graph
        self.nlp = nlp_processor

    def execute(self, context: ExecutionContext):
        # --- Obtención de recursos ---
        try:
            kg = worker_kg
        except NameError:
            if DeepReasoningGene._fallback_kg is None:
                print("[WARN] DeepReasoningGene ejecutándose en proceso principal. Creando instancia de KnowledgeGraph de fallback.")
                DeepReasoningGene._fallback_kg = KnowledgeGraph()
            kg = DeepReasoningGene._fallback_kg

        try:
            nlp = worker_nlp
        except NameError:
            if DeepReasoningGene._fallback_nlp is None:
                print("[WARN] DeepReasoningGene ejecutándose en proceso principal. Creando instancia de spaCy de fallback.")
                DeepReasoningGene._fallback_nlp = spacy.load("es_core_news_sm")
            nlp = DeepReasoningGene._fallback_nlp
        # --- Fin de obtención de recursos ---

        query = context.get("query", "")
        doc = nlp(query)
        
        concepts = [chunk.text.lower() for chunk in doc.noun_chunks]
        concepts = [c for c in concepts if c not in ["qué relación", "que relacion", "relacion", "entre", "y"]]
        
        if len(concepts) < 2:
            context.set("reasoning_result", "Necesito al menos dos conceptos claros en tu pregunta para poder encontrar una relación.")
            return

        concept1, concept2 = concepts[0], concepts[1]
        print(f"  [Gene] DeepReasoning: Analizando la relación entre '{concept1}' y '{concept2}'...")

        # Usa la instancia 'kg' obtenida
        if not kg.has_node(concept1) or not kg.has_node(concept2):
            context.set("reasoning_result", "No tengo suficiente información sobre uno o ambos conceptos para relacionarlos. Quizás una búsqueda web podría ayudar primero.")
            return
            
        try:
            path = kg.find_shortest_path(concept1, concept2)
            if path and 1 < len(path) < 5:
                path_str = " -> ".join(f"'{p}'" for p in path)
                context.set("reasoning_result", f"He encontrado una conexión a través de mi conocimiento: {path_str}.")
                return
        except Exception as e:
            print(f"  [WARN] DeepReasoning: Error buscando camino en KG: {e}")
            pass

        try:
            neighbors1 = set(kg.get_all_neighbors(concept1))
            neighbors2 = set(kg.get_all_neighbors(concept2))
            common_neighbors = neighbors1.intersection(neighbors2)

            if common_neighbors:
                ancestor_str = ", ".join(f"'{a}'" for a in list(common_neighbors)[:3])
                context.set("reasoning_result", f"No están conectados directamente, pero he deducido que ambos son un tipo de o están relacionados con {ancestor_str}.")
                return
        except Exception as e:
            print(f"  [WARN] DeepReasoning: Error buscando ancestros comunes en KG: {e}")
            pass

        context.set("reasoning_result", f"No he podido encontrar una relación clara entre '{concept1}' y '{concept2}' con mi conocimiento actual.")

class DynamicConversationalGene(Gene):
    """
    Genera una respuesta conversacional adaptada al sentimiento
    detectado en la consulta del usuario.
    """
    def execute(self, context: ExecutionContext):
        # Recupera el sentimiento que el gen DetectSentimentGene ya ha guardado
        sentiment = context.get("user_sentiment", "neutral")
        
        print(f"  [Gene] DynamicConversational: Generando respuesta para sentimiento '{sentiment}'.")

        if sentiment == "positive":
            response = "¡Me alegra que lo veas así! Es un tema fascinante y me entusiasma poder conversarlo."
        elif sentiment == "negative":
            response = "Entiendo tu perspectiva. Es un asunto complejo y es normal que genere opiniones encontradas."
        else: # neutral
            response = "Gracias por compartir tu punto de vista. Es un tema interesante sobre el que reflexionar."
            
        context.set_final_response(response)

# **Importante:**
# Revertimos GenerativeResponseGene a su versión original que NO usa un LLM para la síntesis final,
# sino que se basa en la información ya presente en el contexto.
class GenerativeResponseGene(Gene):
    """
    Sintetiza la información del contexto para generar una respuesta
    natural y fluida. Este gen ya no usa un LLM externo para la síntesis final.
    """
    def __init__(self):
        # No necesita llm_client en esta versión
        pass

    def _build_response_from_context(self, context: ExecutionContext) -> str:
        """Crea una respuesta a partir de la información disponible en el contexto."""
        user_query = context.get("query", "la solicitud del usuario")
        
        facts = []
        if summary := context.get("web_summary"):
            facts.append(f"Un resumen de la web dice: '{summary}'")
        if definition := context.get("definition_text"):
            facts.append(f"Mi conocimiento interno define el concepto clave como: '{definition}'")
        if reasoning := context.get("reasoning_result"):
            facts.append(f"Mi análisis de la relación entre conceptos es: '{reasoning}'")
        if calculation := context.get("calculation_result"):
             facts.append(f"El resultado del cálculo es: {calculation}")

        if facts:
            return f"He procesado tu solicitud '{user_query}' y la información relevante es: {'. '.join(facts)}."
        else:
            return f"He procesado tu solicitud '{user_query}', pero no he encontrado información específica para generar una respuesta detallada."

    def execute(self, context: ExecutionContext):
        if context.get_final_response(): # Si otro gen ya dio una respuesta final
            return

        final_text = self._build_response_from_context(context)
        context.set_final_response(final_text)
        print("  [Gene] GenerativeResponse: Respuesta generada a partir del contexto interno.")


# REEMPLAZA la clase RecallLastResponseGene con esta versión
class RecallLastResponseGene(Gene):
    """
    Recupera la última respuesta dada por Prometheus desde la memoria conversacional.
    """
    def __init__(self, mind: 'PrometheusAGI', output_var: str = "recalled_response"):
        self.mind = mind
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        # Esta lógica ahora volverá a funcionar porque self.mind existe
        if self.mind.conversational_history:
            last_response = self.mind.conversational_history[-1].get("prometheus")
            if last_response:
                context.set(self.output_var, f"Lo último que dije fue: '{last_response}'")
                return
        context.set(self.output_var, "No recuerdo haber dicho nada recientemente.")
# Diccionario de operaciones seguras permitidas
operadores_seguros = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
    ast.USub: op.neg
}


class EpisodicMemory:
    """
    Almacena una secuencia de eventos de la "vida" de la IA, ahora con un
    contexto cognitivo completo para el análisis de fallos.
    """
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.log: List[Dict] = []
        self.last_episode_id = 0

    def record_event(self, *, event_type: str, details: Dict):
        """
        Registra un evento de cualquier tipo (acción, percepción, fallo, etc.)
        con una marca de tiempo y un ID.
        """
        if len(self.log) >= self.capacity:
            self.log.pop(0)
        
        self.last_episode_id += 1
        event_record = {
            "id": self.last_episode_id,
            "timestamp": time.time(),
            "type": event_type, # Ej: "ACTION_TAKEN", "GOAL_FAILED"
            "details": details
        }
        self.log.append(event_record)

    def get_events_by_type(self, event_type: str, limit: int = 10) -> List[Dict]:
        """Recupera los últimos N eventos de un tipo específico."""
        return [event for event in reversed(self.log) if event["type"] == event_type][:limit]
    def query_similar_situations(self, current_obs: Dict, top_k: int = 1) -> List[Dict]:
        """
        Encuentra los eventos pasados donde el estado del agente era más similar
        al estado actual, basado en la distancia euclidiana.
        """
        if not self.log or 'agent' not in current_obs:
            return []

        current_agent_pos = np.array(current_obs['agent'])
        
        # Filtramos solo los eventos de acciones pasadas para tener algo que comparar
        action_events = [event for event in self.log if event['type'] == 'ACTION_TAKEN']
        if not action_events:
            return []
            
        distances = []
        for event in action_events:
            # Aseguramos que la observación previa existe y tiene la clave 'agent'
            if 'previous_observation' in event['details'] and 'agent' in event['details']['previous_observation']:
                past_agent_pos = np.array(event['details']['previous_observation']['agent'])
                # Calculamos la distancia euclidiana
                distance = np.linalg.norm(current_agent_pos - past_agent_pos)
                distances.append((distance, event))

        if not distances:
            return []

        # Ordenamos los eventos por distancia (de menor a mayor)
        distances.sort(key=lambda x: x[0])
        
        # Devolvemos los 'top_k' eventos más similares (los de menor distancia)
        return [event for distance, event in distances[:top_k]]


    def get_last_failed_goal_context(self) -> List[Dict]:
        """
        Recupera la secuencia de acciones que precedieron al último fallo de un objetivo.
        """
        try:
            last_failure_index = next(i for i, event in enumerate(reversed(self.log)) if event["type"] == "GOAL_FAILED")
            # Convertir el índice inverso al índice real
            last_failure_index = len(self.log) - 1 - last_failure_index
            # Encontrar el inicio del episodio (el último GOAL_SET o el principio del log)
            start_index = 0
            for i in range(last_failure_index - 1, -1, -1):
                if self.log[i]["type"] in ["GOAL_SET", "EPISODE_START"]:
                    start_index = i
                    break
            return self.log[start_index : last_failure_index + 1]
        except StopIteration:
            return [] # No se encontraron fallos

class CalculatorGene(Gene):
    """
    Un gen que analiza una cadena de texto, extrae una expresión matemática
    y la resuelve de forma segura utilizando Abstract Syntax Trees (AST).
    """
    def _eval_expr(self, expr):
        """Evalúa una expresión de forma segura."""
        return self._eval(ast.parse(expr, mode='eval').body)

    def _eval(self, node):
        if isinstance(node, ast.Num):  # <number>
            return node.n
        elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
            return operadores_seguros[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):  # <operator> <operand>
            return operadores_seguros[type(node.op)](self._eval(node.operand))
        else:
            raise TypeError(node)

    def execute(self, context: ExecutionContext):
        query = context.get("query", "")
        
        # =================================================================
        # INICIO DE LA TRADUCCIÓN DE LENGUAJE NATURAL
        # =================================================================
        # Creamos una copia para no modificar la consulta original
        normalized_query = query.lower()
        # Reemplazamos palabras por operadores
        replacements = {
            " por ": "*",
            " x ": "*",
            " multiplicado por ": "*",
            " dividido entre ": "/",
            " dividido por ": "/",
            " mas ": "+",
            " más ": "+", # Con tilde
            " menos ": "-",
            " elevado a ": "**"
        }
        for word, symbol in replacements.items():
            normalized_query = normalized_query.replace(word, symbol)
        # =================================================================
        
        # Extraemos la expresión matemática de la consulta ya normalizada
        match = re.search(r'([\d\.\+\-\*\/\(\)\s\^]+)', normalized_query)
        if not match:
            context.set("calculation_result", "No encontré una operación matemática válida.")
            return
        
        expression = match.group(1).strip()
        
        try:
            print(f"  [Gene] Calculator: Resolviendo expresión '{expression}' (normalizada desde '{query}')")
            result = self._eval_expr(expression)
            context.set("calculation_result", result)
        except (TypeError, SyntaxError, KeyError, ZeroDivisionError) as e:
            context.set("calculation_result", f"No pude resolver la expresión. Error: {e}")

# EN: la sección de definición de Genes
# REEMPLAZA tu clase AnalizadorGramaticalGene actual con esta:

class AnalizadorGramaticalGene(Gene):
    """
    Usa el procesador de NLP (spaCy) para analizar una frase y devolver un
    desglose de sus componentes gramaticales.
    """
    _fallback_nlp = None # Para uso en el proceso principal

    # El __init__ ya no necesita argumentos
    def __init__(self, nlp_processor):
        self.nlp = nlp_processor

    def execute(self, context: ExecutionContext):
        frase_a_analizar = context.get("query")
        if not frase_a_analizar:
            context.set_final_response("No me has proporcionado una frase para analizar.")
            return

        print(f"  [Gene] AnalizadorGramatical: Analizando la frase '{frase_a_analizar}'...")

        # Intenta usar el 'worker_nlp' global. Si no existe, usa un fallback.
        try:
            doc = worker_nlp(frase_a_analizar)
        except NameError:
            # Fallback si se ejecuta en el proceso principal donde 'worker_nlp' no está definido
            if AnalizadorGramaticalGene._fallback_nlp is None:
                print("[WARN] AnalizadorGramaticalGene ejecutándose en proceso principal. Creando instancia de spaCy de fallback.")
                AnalizadorGramaticalGene._fallback_nlp = spacy.load("es_core_news_sm")
            doc = AnalizadorGramaticalGene._fallback_nlp(frase_a_analizar)

        # El resto de la lógica para crear el reporte permanece igual
        reporte_lineas = ["He analizado la frase y aquí tienes el desglose gramatical:"]
        reporte_lineas.append("-" * 40)
        reporte_lineas.append("PALABRA -> LEMA | CATEGORÍA GRAMATICAL | FUNCIÓN SINTÁCTICA")
        reporte_lineas.append("-" * 40)

        for token in doc:
            linea = f"'{token.text}' -> '{token.lemma_}' | {token.pos_} | {token.dep_}"
            reporte_lineas.append(linea)
        
        reporte_final = "\n".join(reporte_lineas)
        context.set_final_response(reporte_final)
# **Importante:** Eliminamos QueryLLMKnowledgeGene ya que dependía del LLM externo.
# class QueryLLMKnowledgeGene(Gene): ...

# VERIFICA Y REEMPLAZA la clase InferDefinitionFromNeighborsGene con esta versión:

class InferDefinitionFromNeighborsGene(Gene):
    """
    Si un concepto carece de una definición directa, este gen intenta inferir
    una sintetizando las definiciones de sus conceptos vecinos.
    """
    _fallback_nlp = None

    def __init__(self, graph: KnowledgeGraph, nlp_processor):
        self.graph = graph
        self.nlp = nlp_processor

    def execute(self, context: ExecutionContext):
        if context.get("definition_text"):
            return

        topic = context.get("main_topic")
        if not topic:
            return

        print(f"  [Gene] InferDefinition: No hay definición directa para '{topic}'. Intentando inferir a partir de sus vecinos.")

        neighbors = self.graph.get_all_neighbors(topic)
        if not neighbors:
            # ... (lógica existente sin cambios)
            return

        inferred_parts = []
        for neighbor in neighbors[:5]:
            neighbor_def = self.graph.get_node_definition(neighbor)
            if neighbor_def:
                inferred_parts.append(f"Se relaciona con '{neighbor}' que se define como: {neighbor_def}.")
            else:
                inferred_parts.append(f"También se relaciona con '{neighbor}'.")

        if inferred_parts:
            # Esta parte del gen no necesita el nlp_processor, así que no hay que añadirlo.
            inferred_definition = (f"Aunque no tengo una definición directa para '{topic}', "
                                   f"puedo inferir que: {' '.join(inferred_parts)}")
            context.set("definition_text", inferred_definition)
            print(f"  [Gene] InferDefinition: Definición inferida para '{topic}'.")
        else:
            context.set("definition_text", f"No tengo una definición para '{topic}' y no pude inferirla a partir de sus conceptos relacionados.")

# REEMPLAZA tu clase LearnFromTextGene actual con esta versión completa y corregida:

class LearnFromTextGene(Gene):
    """
    Procesa un bloque de texto de una variable de contexto específica,
    extrae conceptos clave y los integra en el KnowledgeGraph.
    Es compatible con el modo de multiprocessing.
    """
    # Atributos de clase para guardar las instancias de fallback en el proceso principal
    _fallback_nlp = None
    _fallback_kg = None

    def __init__(self, text_var: str, graph: KnowledgeGraph, nlp_processor, similarity_model):
        self.text_var = text_var
        self.graph = graph
        self.nlp = nlp_processor
        self.similarity_model = similarity_model

    def execute(self, context: ExecutionContext):
        # --- Obtención de recursos ---
        # Intenta usar los recursos globales del worker.
        # Si no existen, crea y usa una instancia de fallback para el proceso principal.
        try:
            # Usa el KnowledgeGraph del worker
            kg = worker_kg
        except NameError:
            # Fallback para cuando se ejecuta fuera de un worker (ej. modo interactivo)
            if LearnFromTextGene._fallback_kg is None:
                print("[WARN] LearnFromTextGene ejecutándose en proceso principal. Creando instancia de KnowledgeGraph de fallback.")
                LearnFromTextGene._fallback_kg = KnowledgeGraph()
            kg = LearnFromTextGene._fallback_kg

        try:
            # Usa el modelo spaCy del worker
            nlp = worker_nlp
        except NameError:
            # Fallback para spaCy
            if LearnFromTextGene._fallback_nlp is None:
                print("[WARN] LearnFromTextGene ejecutándose en proceso principal. Creando instancia de spaCy de fallback.")
                LearnFromTextGene._fallback_nlp = spacy.load("es_core_news_sm")
            nlp = LearnFromTextGene._fallback_nlp
        # --- Fin de obtención de recursos ---

        text_to_learn = context.get(self.text_var)
        main_topic = context.get("main_topic")

        if not text_to_learn or not main_topic:
            return

        print(f"  [Gene] LearnFromText: Procesando texto de '{self.text_var}' para actualizar mi conocimiento sobre '{main_topic}'...")
        
        doc = nlp(text_to_learn)
        sentences = [sent.text.strip() for sent in doc.sents]
        definition = " ".join(sentences[:2])

        # Usa la instancia de KnowledgeGraph obtenida (sea del worker o del fallback)
        kg.add_node_with_definition(main_topic, definition)
        
        related_concepts = set(chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) < 4)

        for concept in related_concepts:
            if concept != main_topic:
                kg.add_node_if_not_exists(concept)
                kg.add_edge(main_topic, concept, "related_to")

        print(f"  [Gene] LearnFromText: Conocimiento sobre '{main_topic}' integrado. Se crearon {len(related_concepts)} conexiones.")

class SetVariableGene(Gene):
    def __init__(self, key: str, value: any): self.key, self.value = key, value
    def execute(self, context: ExecutionContext): context.set(self.key, self.value)


# REEMPLAZA tu clase GetNodeDefinitionGene actual con esta versión completa y corregida:

class GetNodeDefinitionGene(Gene):
    """
    Busca una definición en el Knowledge Graph.
    Es compatible con el modo de multiprocessing.
    """
    # Atributo de clase para guardar la instancia de fallback
    _fallback_kg = None

    def __init__(self, entity_var: str, graph: KnowledgeGraph):
        self.entity_var = entity_var
        self.graph = graph

    def execute(self, context: ExecutionContext):
        # --- Obtención de recursos ---
        try:
            kg = worker_kg
        except NameError:
            if GetNodeDefinitionGene._fallback_kg is None:
                print("[WARN] GetNodeDefinitionGene ejecutándose en proceso principal. Creando instancia de KnowledgeGraph de fallback.")
                GetNodeDefinitionGene._fallback_kg = KnowledgeGraph()
            kg = GetNodeDefinitionGene._fallback_kg
        # --- Fin de obtención de recursos ---

        concept_to_define = None
        entities = context.get(self.entity_var, [])
        if entities and isinstance(entities, list) and len(entities) > 0:
            concept_to_define = str(entities[0]).lower()
        if not concept_to_define:
            topic = context.get("main_topic")
            if topic:
                concept_to_define = str(topic).lower()

        if not concept_to_define:
            context.set("definition_text", "No se especificó un concepto para definir.")
            return

        print(f"  [Gene] GetNodeDefinition: Buscando definición para el concepto '{concept_to_define}' en Neo4j.")
        
        # Usa la instancia de KnowledgeGraph obtenida
        definition = kg.get_node_definition(concept_to_define)
        if definition:
            print(f"  [Gene] GetNodeDefinition: Definición explícita encontrada en la base de datos.")
            context.set("definition_text", definition)
            return

        print(f"  [Gene] GetNodeDefinition: No se encontró definición explícita para '{concept_to_define}'.")
        context.set("definition_text", None)

class FormulateResponseGene(Gene):
    """
    Un gen de respuesta final mejorado. Escanea el contexto en busca de resultados
    clave (resúmenes, definiciones, etc.) y formula una respuesta final con el
    primero que encuentra.
    """
    def __init__(self, fallback_message: str = "He procesado la solicitud, pero no he generado una respuesta textual."):
        # Ya no necesita 'input_var'
        self.fallback_message = fallback_message
        # Lista de claves de contexto a buscar, en orden de prioridad
        self.result_keys_in_priority = [
            "definition_text",
            "scientific_summary",
            "web_summary",
            "reasoning_result",
            "calculation_result",
            "syntax_analysis_report",
            "plan_explanation",
            "patterns",
            "life_story"
        ]

    def execute(self, context: ExecutionContext):
        # Si otro gen ya dio una respuesta final (ej. AnalizadorGramatical), la respetamos.
        if context.get_final_response():
            return

        # Busca en el contexto la primera variable de resultado que contenga información.
        response_text = None
        for key in self.result_keys_in_priority:
            if found_text := context.get(key):
                response_text = str(found_text) # Aseguramos que sea un string
                print(f"  [Gene] FormulateResponse: Respuesta encontrada en la variable de contexto '{key}'.")
                break
        
        # Si encontró algo, esa es la respuesta. Si no, usa el mensaje de fallback.
        context.set_final_response(response_text if response_text else self.fallback_message)

class EvolveStrategyGene(Gene):
    """
    Un "meta-gen" que activa un ciclo de evolución interno (un "Dojo") 
    para forjar una estrategia óptima para una consulta compleja en tiempo real.
    """
    # MODIFICACIÓN: Aceptar 'mind' en lugar de 'dojo' para romper la dependencia circular.
    def __init__(self, mind: 'PrometheusAGI', generations: int = 10, population_size: int = 20):
        self.mind = mind # Guardar la referencia a la mente principal
        self.generations = generations
        self.population_size = population_size
        print(f"[Gene] EvolveStrategyGene inicializado. ¡Listo para el pensamiento profundo!")

    def execute(self, context: ExecutionContext):
        query = context.get("query")
        intent = context.get("intent")

        if not query or not intent:
            context.set("evolved_answer", "No puedo evolucionar una estrategia sin una consulta y una intención claras.")
            return

        print(f"\n===== [EvolveStrategyGene] INICIANDO DOJO INTERNO PARA LA CONSULTA: '{query[:50]}...' =====")
        context.log_thought(
            "EvolveStrategyGene", 
            f"La consulta es compleja. Activando ciclo de evolución de {self.generations} generaciones para forjar una respuesta óptima."
        )

        web_search_ctx = ExecutionContext(initial_vars={"main_topic": query, "query": query})
        WebSearchGene(query_var="main_topic", output_key="web_summary").execute(web_search_ctx)
        ideal_response_base = web_search_ctx.get("web_summary", "")

        if not ideal_response_base or len(ideal_response_base) < 100:
            print("[EvolveStrategyGene] No se pudo obtener una base de conocimiento sólida de la web. La evolución será puramente exploratoria.")
            ideal_response_base = query

        # MODIFICACIÓN: Acceder a 'dojo' a través de 'self.mind' en tiempo de ejecución.
        best_chromosome = self.mind.dojo._evolve_on_challenge(
            instruction=query,
            ideal_response=ideal_response_base,
            intent=intent,
            generations=self.generations,
            population_size=self.population_size
        )

        if best_chromosome:
            print("===== [EvolveStrategyGene] DOJO INTERNO COMPLETADO. Ejecutando la estrategia optimizada... =====")
            final_context = ExecutionContext(initial_vars=context.memory)
            
            for gene in best_chromosome.genes:
                gene.execute(final_context)
            
            final_answer = final_context.get_final_response()
            
            context.set_final_response(final_answer)
            context.log_thought(
                "EvolveStrategyGene", 
                f"Evolución completada. La mejor estrategia encontrada (Fitness: {best_chromosome.fitness:.2f}) ha generado la respuesta final."
            )
        else:
            context.set_final_response("Tras un proceso de reflexión interna, no he podido encontrar una estrategia satisfactoria para tu pregunta.")

class DetectSentimentGene(Gene):
    def execute(self, context: ExecutionContext):
        q = context.get("query", "").lower()
        if any(w in q for w in ["fatal", "odio", "triste"]): s = "negative"
        elif any(w in q for w in ["gracias", "genial", "perfecto"]): s = "positive"
        else: s = "neutral"
        context.set("user_sentiment", s)

class TellMyStoryGene(Gene):
    """Activa la capacidad de la IA para contar la historia de su vida."""
    def __init__(self, narrative_self, output_var: str = "life_story"):
        self.narrative_self = narrative_self
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        story = self.narrative_self.get_life_summary()
        context.set(self.output_var, story)


class StyleAnalyzerGene(Gene):
    def execute(self, context: ExecutionContext):
        text = context.get("query", "").lower()
        formality = "informal" if any(w in text for w in ["tío", "osea", "que tal"]) else "formal"
        context.set("user_style_profile", {"formality": formality})

class StylizedResponseGene(Gene):
    def __init__(self, intent: str): self.intent = intent
    def execute(self, context: ExecutionContext):
        style = context.get("user_style_profile", {"formality": "formal"})
        options = LEXICON_BRAIN_V2.get(self.intent, [])
        if options:
            matches = [o for o in options if o["style"]["formality"] == style["formality"]]
            context.set_final_response(random.choice(matches if matches else options)["text"])

import google.generativeai as genai
import re
import subprocess
class VoiceInteractionGene(Gene):
    """
    Gen que permite a Prometheus escuchar al usuario y responder con voz.
    Guarda la transcripción en el contexto bajo 'query'.
    Puede hablar el texto final almacenado en 'final_answer_text'.
    """

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 160)  # Velocidad de habla

    def listen(self) -> str:
        with sr.Microphone() as source:
            print("[🎧] Escuchando...")
            
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio, language="es-ES")
            print(f"[👤 Usuario]: {text}")
            return text
        except sr.UnknownValueError:
            self.speak("No entendí lo que dijiste.")
            return ""
        except sr.RequestError:
            self.speak("Hubo un error al conectar con el servicio de voz.")
            return ""

    def speak(self, text: str):
        print(f"[🗣️ Prometheus]: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def execute(self, context: ExecutionContext):
        # Escucha al usuario
        user_input = self.listen()
        context.set("query", user_input)

        # Si ya hay una respuesta generada, la habla
        response = context.get_final_response()
        if response:
            self.speak(response)

class LLMCodeGenerator:
    """
    Se comunica con la API de Google Gemini para generar y probar código de genes.
    """
    def __init__(self, api_key: str):
        if not api_key:
            print("[WARN] LLMCodeGenerator: La variable de entorno GOOGLE_API_KEY no está configurada. La génesis de genes no funcionará.")
            self.model = None
            return
        
        try:
            genai.configure(api_key=api_key)
            # Usamos un modelo potente, ideal para razonamiento y generación de código.
            self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
            print("[LLM_CODE_GEN] Conectado a Gemini 1.5 Pro.")
        except Exception as e:
            print(f"[ERROR] No se pudo configurar la API de Gemini: {e}")
            self.model = None

    def _extract_python_code(self, text: str) -> str:
        """
        Extrae el bloque de código Python de la respuesta del LLM.
        Busca bloques delimitados por ```python ... ```.
        """
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback por si el LLM solo devuelve el código plano
        return text.strip()

    def generate_code(self, prompt: str) -> str:
        """
        Envía un prompt a Gemini y extrae el código Python de la respuesta.
        """
        if not self.model: return ""
        print("[LLM_CODE_GEN] Enviando prompt a Gemini para generación de gen...")
        
        try:
            response = self.model.generate_content(prompt)
            print("[LLM_CODE_GEN] Respuesta recibida de Gemini.")
            return self._extract_python_code(response.text)
        except Exception as e:
            print(f"[ERROR] Falló la llamada a la API de Gemini para generar código: {e}")
            return ""

    def generate_test_code(self, gene_code: str, intent: str, sample_query: str) -> str:
        """
        Genera el código de una función de prueba para un gen dado.
        """
        if not self.model: return "def test_case(): return False"
        print("[LLM_CODE_GEN] Enviando prompt a Gemini para generación de test...")
        
        prompt = f"""
Basado en la siguiente clase de Gen de Python:
--- CÓDIGO DEL GEN ---
{gene_code}
--- FIN DEL CÓDIGO ---

La intención original era '{intent}' y una consulta de ejemplo fue '{sample_query}'.

Escribe una única función de Python llamada `test_case()` que haga lo siguiente:
1. Cree una instancia de la clase del gen.
2. Cree un objeto `ExecutionContext`.
3. Configure el `ExecutionContext` con los datos de entrada necesarios para el gen.
4. Ejecute el método `execute` del gen.
5. Use una o más sentencias `assert` para verificar que el resultado en el contexto es el esperado.
6. La función debe imprimir información útil sobre lo que está probando.
7. Si todas las aserciones pasan, la función debe devolver `True`.

No incluyas nada más que la función `test_case()` en tu respuesta.
"""
        try:
            response = self.model.generate_content(prompt)
            print("[LLM_CODE_GEN] Código de prueba recibido de Gemini.")
            return self._extract_python_code(response.text)
        except Exception as e:
            print(f"[ERROR] Falló la llamada a la API de Gemini para generar el test: {e}")
            return "def test_case(): return False"
        
class GeneForge:
    """
    La forja de genes de Prometheus. Un sistema de auto-expansión que puede
    diseñar, generar, probar e integrar nuevas capacidades (Genes) en tiempo de ejecución.
    """
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind
        self.genesis_corpus_path = self.mind.paths.get("genesis_corpus", "gene_forge/genesis_corpus.jsonl")
        
        # Oráculo Externo (Maestro), para aprender nuevas soluciones.
        self.external_oracle = LLMCodeGenerator(api_key=GOOGLE_API_KEY)
        os.makedirs(FORGE_DIR, exist_ok=True)
        # Asegurarse de que los directorios necesarios existen.
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        os.makedirs(CUSTOM_GENES_DIR, exist_ok=True)

    def _create_gene_generation_prompt(self, intent, research, gene_name) -> str:
        """Crea un prompt detallado y bien estructurado para guiar al LLM."""
        return f"""
Actúa como un programador de élite diseñando un componente modular (un "Gen") para una IA evolutiva llamada Prometheus.

**TAREA:**
Escribe una clase de Python completa llamada `{gene_name}`. Esta clase debe resolver la incapacidad de la IA para manejar la intención: '{intent}'.

**CONTEXTO DE LA ARQUITECTURA:**
Un "Gen" es una clase que hereda de `Gene` y su lógica principal reside en el método `execute`. Los genes interactúan entre sí a través de un `ExecutionContext`.

Aquí están las definiciones base que DEBES usar:
```python
class Gene:
    def execute(self, context: 'ExecutionContext'): raise NotImplementedError

class ExecutionContext:
    def __init__(self, initial_vars=None): 
        self.memory = initial_vars or {{}}
    def set(self, key, value): 
        self.memory[key] = value
    def get(self, key, default=None): 
        return self.memory.get(key, default)
```

**REQUISITOS:**
1. La clase debe heredar de `Gene`
2. Implementar el método `execute(self, context: ExecutionContext)`
3. Usar context.get() para leer variables y context.set() para escribir resultados
4. Incluir manejo de errores básico
5. Añadir comentarios explicativos

**INVESTIGACIÓN DE CONTEXTO:**
{research}

**RESPUESTA ESPERADA:**
Solo el código Python de la clase, sin explicaciones adicionales.
"""

    def _save_successful_genesis(self, prompt: str, code: str):
        """Añade un ejemplo exitoso de prompt->código al corpus de entrenamiento."""
        try:
            with open(self.genesis_corpus_path, "a", encoding="utf-8") as f:
                record = {"prompt": prompt, "completion": code}
                f.write(json.dumps(record) + "\n")
            print(f"[FORGE] Lección de génesis guardada en '{self.genesis_corpus_path}'.")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar la lección de génesis: {e}")

    def _assemble_gene_file(self, gene_name: str, gene_code: str) -> str:
        """
        Toma el código del gen generado y lo envuelve en un archivo Python completo
        y ejecutable dentro del sandbox para su prueba.
        """
        temp_file_path = os.path.join(SANDBOX_DIR, f"temp_{gene_name}.py")
        
        # Código de plantilla que importa las dependencias base necesarias.
        full_code = f"""# Archivo de gen generado automáticamente para pruebas en el sandbox.
import sys
import os

# Añadimos el directorio raíz al path para que pueda encontrar las clases base.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importaciones mínimas para que un gen sea válido.
from PROMETEUSV5 import Gene, ExecutionContext

# El código del gen generado por el LLM.
{gene_code}
"""
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)

        print(f"[FORGE] Archivo temporal del gen ensamblado en: '{temp_file_path}'")
        return temp_file_path

    def _test_in_sandbox(self, gene_path: str, test_code: str) -> bool:
        """
        Ejecuta el código de prueba generado por el LLM en un entorno seguro
        para validar la funcionalidad del nuevo gen.
        """
        if not test_code:
            print("[FORGE_TEST] El LLM no generó código de prueba. Fallo automático.")
            return False

        test_path = os.path.join(SANDBOX_DIR, "temp_test_case.py")
        
        with open(gene_path, "r", encoding="utf-8") as f:
            gene_full_code = f.read()
            
        full_test_script = f"""{gene_full_code}

# --- Código de Prueba Generado por el LLM ---
{test_code}

# --- Ejecución de la Prueba ---
if __name__ == "__main__":
    try:
        result = test_case()
        if result:
            print("TEST_PASSED")
            sys.exit(0)  # Salida exitosa
        else:
            print("TEST_FAILED: test_case() returned False")
            sys.exit(1)  # Salida con error
    except Exception as e:
        import traceback
        print(f"TEST_EXCEPTION: {e}")
        traceback.print_exc()
        sys.exit(1)  # Salida con error
"""
        
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(full_test_script)

        try:
            print("[FORGE_TEST] Ejecutando prueba de gen en un subproceso seguro...")
            # Usamos subprocess para aislar completamente la ejecución.
            result = subprocess.run(
                [sys.executable, test_path], 
                capture_output=True, text=True, timeout=30
            )
            print(f"[FORGE_TEST] Salida de la prueba:\n---INICIO---\n{result.stdout}\n{result.stderr}\n---FIN---")
            
            return result.returncode == 0 and "TEST_PASSED" in result.stdout
        except subprocess.TimeoutExpired:
            print("[FORGE_TEST] La prueba excedió el tiempo límite. Fallo automático.")
            return False
        except Exception as e:
            print(f"[FORGE_TEST] Ocurrió un error inesperado al ejecutar la prueba: {e}")
            return False

    def _integrate_gene(self, temp_gene_path: str, gene_name: str):
        """
        Mueve el archivo del gen desde el sandbox al directorio de genes personalizados
        para que pueda ser utilizado en el futuro.
        """
        final_path = os.path.join(CUSTOM_GENES_DIR, f"{gene_name}.py")
        os.rename(temp_gene_path, final_path)
        print(f"[FORGE] Gen '{gene_name}' movido a '{final_path}'.")
        print("[FORGE] ¡Integración completada! El nuevo gen estará disponible en el próximo reinicio.")

    def attempt_new_gene_creation(self, intent: str, sample_query: str):
        """
        El método principal que orquesta el ciclo completo de creación de un nuevo gen.
        """
        print(f"\n===== CICLO DE GÉNESIS DE GENES INICIADO PARA LA INTENCIÓN '{intent}' =====")
        
        research_summary = f"La investigación sobre la intención '{intent}' sugiere la necesidad de una solución programática específica. Consulta de ejemplo: '{sample_query}'"
        gene_name = f"{intent.capitalize()}Gene"
        prompt = self._create_gene_generation_prompt(intent, research_summary, gene_name)

        print("[FORGE] Consultando al Oráculo Externo (Gemini)...")
        gene_code = self.external_oracle.generate_code(prompt)

        if not gene_code:
            print("[FORGE] El oráculo no pudo generar código. Abortando génesis.")
            return

        print("[FORGE] Código del gen generado por el Oráculo Externo (Gemini).")
        
        # Ensamblar, probar e integrar el gen
        temp_gene_path = self._assemble_gene_file(gene_name, gene_code)
        test_code = self.external_oracle.generate_test_code(gene_code, intent, sample_query)
        test_passed = self._test_in_sandbox(temp_gene_path, test_code)

        if test_passed:
            print(f"[FORGE] ¡Prueba superada! Integrando '{gene_name}' en el genoma de Prometheus...")
            self._save_successful_genesis(prompt, gene_code)
            self._integrate_gene(temp_gene_path, gene_name)
        else:
            print("[FORGE] La prueba del gen falló. Descartando el gen.")
            
        print("====================== GÉNESIS DE GENES FINALIZADO ======================\n")
class WebSearchGene(Gene):
    """
    Gen de búsqueda web que utiliza DuckDuckGo. Versión robusta.
    """
    def __init__(self, query_var: str = "query", output_key: str = "web_summary", sentences_in_summary: int = 3):
        self.query_var = query_var
        self.output_key = output_key
        self.sentences_in_summary = sentences_in_summary

    def _perform_duckduckgo_search(self, query_text: str) -> list[str]:
        search_url = f"https://html.duckduckgo.com/html/?q={query_text.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        print(f"  [WebSearchGene] Realizando scraping en: {search_url}")
        try:
            response = requests.get(search_url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select('a.result__a')
            urls = [link['href'] for link in links if link.has_attr('href')]
            
            if not urls:
                print("  [WebSearchGene] ADVERTENCIA: No se encontraron enlaces de resultados.")
            else:
                print(f"  [WebSearchGene] Se encontraron {len(urls)} enlaces.")
            return urls[:3]
        except Exception as e:
            print(f"  [WebSearchGene] ERROR: Falló el scraping en DuckDuckGo: {e}")
            return []

    def _scrape_and_clean_url(self, url: str) -> str | None:
        try:
            print(f"    -> Scrapeando URL del resultado: {url[:80]}")
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            raw_text = ' '.join(p.get_text() for p in paragraphs)
            cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
            return cleaned_text if len(cleaned_text) > 200 else None
        except Exception as e:
            print(f"    -> ERROR al procesar URL {url[:80]}: {e}")
            return None

    def execute(self, context: ExecutionContext):
        query_text = context.get(self.query_var)
        if not query_text:
            context.set(self.output_key, None)
            return

        urls = self._perform_duckduckgo_search(query_text)
        if not urls:
            context.set(self.output_key, None)
            return

        all_clean_texts = [text for url in urls if (text := self._scrape_and_clean_url(url))]
        
        if all_clean_texts:
            combined_text = " ".join(all_clean_texts)
            parser = PlaintextParser.from_string(combined_text, Tokenizer("spanish"))
            summarizer = LsaSummarizer()
            summary_sentences = summarizer(parser.document, self.sentences_in_summary)
            final_summary = " ".join(str(s) for s in summary_sentences)
            context.set(self.output_key, final_summary)
            print(f"  [WebSearchGene] Resumen generado con éxito ({len(final_summary)} caracteres).")
        else:
            context.set(self.output_key, None)
            print("  [WebSearchGene] ADVERTENCIA: Se encontraron enlaces, pero no se pudo extraer contenido útil.")
# #############################################################################
# PARTE 3: GÉNESIS AUTÓNOMO
# #############################################################################
# Mapeo de nombres de clase para el intérprete (debe ir después de la definición de WebSearchGene)
AVAILABLE_GENES = {
    "WebSearchGene": WebSearchGene
}

class DynamicIntentEngine:
    """
    Analiza una consulta para extraer su intención principal y las entidades clave.
    Esta versión tiene una lógica de extracción de tópicos mejorada y anulación de intención.
    """
    def __init__(self):
        print("[INTENT_ENGINE] Motores de NLU listos para carga perezosa.")
        self._classification_pipeline = None
        self._nlp = None
        self.candidate_intents = {
            "DEFINIR": "pregunta por la definición o el significado de algo",
            "BUSCAR_WEB": "petición para investigar información en internet",
            "BUSCAR_CIENCIA": "petición para buscar artículos científicos o papers",
            "RELACIONAR": "pregunta por la relación entre dos o más cosas",
            "ANALIZAR_ESTRATEGIAS": "petición para analizar cómo resuelve un problema",
            "ANALIZAR_GRAMATICA": "petición para analizar una frase",
            "ANALIZAR_CODIGO": "petición para explicar el código de un gen",
            "CONVERSACIONAL": "una pregunta o comentario general",
            "CALCULAR": "petición para resolver una operación matemática",
            "RECORDAR": "petición para recordar algo dicho anteriormente",
        }
        self.candidate_labels = list(self.candidate_intents.keys())

    @property
    def classification_pipeline(self):
        """Carga el pipeline de clasificación solo cuando se accede por primera vez."""
        if self._classification_pipeline is None:
            print("\n[INTENT_ENGINE] Cargando modelo de clasificación zero-shot (primera vez)...")
            self._classification_pipeline = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        return self._classification_pipeline

    @property
    def nlp(self):
        """Carga el modelo de spaCy solo cuando se accede por primera vez."""
        if self._nlp is None:
            print("\n[INTENT_ENGINE] Cargando modelo 'es_core_news_sm' de spaCy (primera vez)...")
            self._nlp = spacy.load("es_core_news_sm")
        return self._nlp

    def _get_main_topic(self, text: str, intent: str) -> str:
        """
        Versión final y robusta para extraer el tópico principal, limpiando frases de comando.
        """
        text_lower = text.lower().strip()
        
        # Lista de prefijos de comando a eliminar para aislar el sujeto
        prefixes_to_strip = {
            'DEFINIR': [
                "define el concepto de", "define la idea de", "define", 
                "qué es", "que es", "cuál es el significado de", "significado de"
            ],
            'BUSCAR_WEB': ["busca en internet sobre", "busca sobre", "investiga sobre", "encuentra información de"],
            'BUSCAR_CIENCIA': ["busca un paper sobre", "encuentra artículos de", "investigación científica sobre"],
            'ANALIZAR_CODIGO': ["analiza el código de", "analiza el gen"],
        }
        
        # Si la intención tiene prefijos definidos, los eliminamos
        if intent in prefixes_to_strip:
            for prefix in prefixes_to_strip[intent]:
                if text_lower.startswith(prefix):
                    # Elimina el prefijo y limpia espacios sobrantes
                    topic = text_lower[len(prefix):].strip()
                    # Limpia comillas o caracteres extraños y devuelve el resultado
                    return topic.strip("?¿ '\"")
        
        # Si no hubo coincidencia de prefijo, usamos un método de fallback más general
        doc = self.nlp(text)
        # Priorizamos los trozos de sustantivos (noun chunks)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ == "NOUN"]
        if noun_chunks:
            # Devuelve el trozo de sustantivo más largo, que suele ser el más específico
            return max(noun_chunks, key=len)

        return text # Devolvemos el texto original si todo lo demás falla

    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Procesa la consulta del usuario, determina la intención y extrae el tópico limpio.
        """
        query_lower = query.lower()
        forced_intent = None

        # Lógica de anulación por palabras clave para mejorar la precisión en comandos comunes
        intent_keywords = {
            "SALUDAR": ["hola", "buenas", "buenos días", "qué tal"],
            "RECORDAR": ["repite", "repíteme", "qué dijiste", "lo anterior"],
            "BUSCAR_CIENCIA": ["busca un paper", "artículo científico", "pubmed"],
            "ANALIZAR_CODIGO": ["analiza el código", "explícame el gen"],
            "DEFINIR": ["qué es", "define", "significado de"],
            "CALCULAR": ["cuánto es", "calcula", "suma", "resta", "multiplica", "divide"],
        }
        for intent_key, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                forced_intent = intent_key
                print(f"  [Intent Override] Forzando la intención a '{forced_intent}'.")
                break
        
        if forced_intent:
            intent = forced_intent
            confidence = 1.0
        else:
            # Si no hay anulación, usamos el modelo de clasificación
            analysis_result = self.classification_pipeline(query, self.candidate_labels, hypothesis_template="Esta frase es una {}.")
            intent = analysis_result['labels'][0]
            confidence = analysis_result['scores'][0]

        # Usamos nuestro método de extracción de tópico mejorado
        main_topic = self._get_main_topic(query, intent)
        
        print(f"  [Intent Analysis] Query: '{query[:30]}...' -> Intent: {intent} ({confidence:.2f}), Topic: '{main_topic}'")

        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]

        return {"intent": intent, "confidence": confidence, "main_topic": main_topic, "entities": entities, "full_query": query}
    
class AutonomousGenesisEngine:
    def __init__(self, kg: KnowledgeGraph, percentile: int):
        self.kg, self.percentile = kg, percentile
        try:
            self.nlp_es = spacy.load("es_core_news_sm", disable=["parser"])
        except OSError:
            print("Modelo de spaCy 'es_core_news_sm' no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
            raise
    def run(self, corpus: List[Dict[str, str]]):
        print("[GÉNESIS] Extrayendo conceptos y relaciones del corpus...")
        all_text = " ".join(item["INSTRUCTION"] + " " + item["RESPONSE"] for item in corpus)
        doc = self.nlp_es(all_text)
        entities = [chunk.text for chunk in doc.noun_chunks if len(chunk.text.split()) < 4]
        entity_counts = Counter(entities)
        
        top_entities = [entity for entity, count in entity_counts.most_common(1000)]
        print(f"[GÉNESIS] {len(top_entities)} conceptos clave descubiertos.")
        
        for entity in top_entities:
            definition = "Definición no encontrada en el corpus."
            for item in corpus:
                if entity in item["INSTRUCTION"].lower():
                    definition = item["RESPONSE"]
                    break
            self.kg.add_node_if_not_exists(entity, label=entity, definition=definition)
        
        print("--- GÉNESIS COMPLETADO ---")
        return {e: re.compile(r"\b" + re.escape(e) + r"\b", re.IGNORECASE) for e in top_entities}

# #############################################################################
# PARTE 4 & 5: FORJA Y METACOGNICIÓN
# #############################################################################


class PopulationGenerator:
    """
    Genera poblaciones de cromosomas.
    Esta versión delega la creación de estrategias semilla al StrategicPlanner,
    eliminando la necesidad de una lógica de casos codificada.
    """
    def __init__(self, mind: 'PrometheusAGI', intent_engine: 'DynamicIntentEngine', strategic_planner: 'StrategicPlanner'):
        """
        El inicializador ahora acepta el StrategicPlanner.
        """
        self.mind = mind
        self.intent_engine = intent_engine
        self.strategic_planner = strategic_planner
        self.full_arsenal = mind.full_arsenal

    def generate(self, query: str, size: int) -> List['Chromosome']:
        """
        Genera una población de cromosomas usando especialistas existentes
        y el planificador dinámico para crear nuevas estrategias.
        """
        print(f"\n[PopGen] Generando población para la consulta: '{query[:50]}...'")
        analysis = self.intent_engine.analyze(query)
        intent = analysis['intent']
        
        pop: List[Chromosome] = []
        
        print(f"  [PopGen] Buscando especialistas para la intención: '{intent}'...")
        specialists = self.mind.get_specialists_for_intent(intent)
        if specialists:
            print(f"  [PopGen] {len(specialists)} especialista(s) encontrado(s). Usándolos como base.")
            pop.extend([copy.deepcopy(s) for s in specialists])

        print("  [PopGen] Solicitando nueva estrategia al StrategicPlanner...")
        seed_chromosome = self.strategic_planner.plan_seed_strategy(query, analysis)
        pop.append(seed_chromosome)

        while len(pop) < size:
            if pop:
                parent_to_mutate = max(pop, key=lambda c: c.fitness if c.fitness is not None else -1)
                mutated_child = copy.deepcopy(parent_to_mutate)
                
                # --- CORRECCIÓN ---
                # Convertimos los valores del diccionario del arsenal en una lista
                # para que random.choice() pueda funcionar correctamente.
                arsenal_list = list(self.full_arsenal.values())
                temp_evolver = EvolutionEngine(arsenal_list, 0, {"swap": 0.5, "insertion": 0.5})
                # --- FIN DE LA CORRECCIÓN ---
                
                mutated_child = temp_evolver._mutate(mutated_child)
                mutated_child.description += " (Hijo Mutado)"
                pop.append(mutated_child)

            else:
                random_thought = random.choices(self.full_arsenal, k=random.randint(1, 3))
                pop.append(Chromosome(random_thought, "Hijo del Caos (Fallback)"))
                
        return pop[:size]
    
class FitnessCalculator:
    """
    Calcula un conjunto de métricas de rendimiento ("raw scores") para un cromosoma dado.
    Esta versión separa el cálculo de la novedad para un manejo correcto en entornos paralelos.
    """
    def __init__(self, sim_model: SentenceTransformer):
        self.model = sim_model

    def calculate_novelty_score(self, actual_response: str, other_responses: List[str]) -> float:
        """
        Calcula únicamente la métrica de novedad comparando una respuesta con una lista de otras.
        Este método se llama desde el proceso principal que tiene la visión completa de la población.
        """
        if not actual_response or not other_responses:
            return 0.0
        try:
            emb_p = self.model.encode(actual_response, convert_to_tensor=True)
            other_embs = self.model.encode(other_responses, convert_to_tensor=True)
            similarities = util.pytorch_cos_sim(emb_p, other_embs)[0]
            avg_similarity = torch.mean(similarities).item()
            return 1.0 - avg_similarity
        except Exception as e:
            return 0.0
    
    def calculate_structural_novelty(self, chromosome: 'Chromosome', population_chromosomes: List['Chromosome']) -> float:
        """
        Calcula una métrica de novedad barata basada en la estructura del cromosoma (los genes que lo componen).
        Compara cuán diferente es este cromosoma de los demás en la población.
        """
        if not population_chromosomes:
            return 1.0

        current_gene_set = set(g.__class__.__name__ for g in chromosome.genes)
        
        total_jaccard_distance = 0
        
        for other_chromosome in population_chromosomes:
            if other_chromosome == chromosome: continue
            
            other_gene_set = set(g.__class__.__name__ for g in other_chromosome.genes)
            
            intersection_size = len(current_gene_set.intersection(other_gene_set))
            union_size = len(current_gene_set.union(other_gene_set))
            
            if union_size == 0:
                jaccard_similarity = 1.0
            else:
                jaccard_similarity = intersection_size / union_size
            
            total_jaccard_distance += (1.0 - jaccard_similarity)
            
        return total_jaccard_distance / len(population_chromosomes)
    
    def calculate_metrics(self, chromosome: 'Chromosome', ideal_response: str, gene_usage: Counter) -> Dict[str, float]:
        """
        Calcula las métricas que NO dependen de otras respuestas de la población (es decir, todas excepto la novedad).
        Este método es llamado por el trabajador paralelo.
        """
        if not chromosome.final_context or chromosome.final_context.get("execution_error"):
            return {"similarity": 0, "efficiency": -100, "detail": 0, "curiosity": 0}

        actual_response = chromosome.final_context.get_final_response()
        if not actual_response:
            return {"similarity": 0, "efficiency": -len(chromosome.genes), "detail": 0, "curiosity": 0}

        # 1. Similitud (precisión)
        similarity_score = 0.0
        if ideal_response:
            try:
                emb_p = self.model.encode(actual_response, convert_to_tensor=True)
                emb_m = self.model.encode(ideal_response, convert_to_tensor=True)
                similarity_score = util.pytorch_cos_sim(emb_p, emb_m).item()
            except Exception:
                similarity_score = 0.0

        # 2. Eficiencia (coste)
        efficiency_score = -len(chromosome.genes)

        # 3. Detalle (verbosidad)
        detail_score = len(actual_response)

        # 4. Curiosidad (exploración)
        curiosity_score = 0.0
        total_uses = sum(gene_usage.values()) if gene_usage else 0
        if total_uses > 0:
            for gene in chromosome.genes:
                rarity = 1.0 - (gene_usage.get(gene.__class__.__name__, 0) / total_uses)
                curiosity_score += rarity
        
        return {
            "similarity": similarity_score * 1000,
            "efficiency": efficiency_score,
            "detail": detail_score,
            "curiosity": curiosity_score
        }
# REEMPLAZA la clase ChromosomeExecutor completa con esta versión:

class ChromosomeExecutor:
    """
    Ejecuta los genes de un cromosoma en secuencia.
    Versión final, consciente del entorno asíncrono.
    """
    @staticmethod
    async def execute_async(chromosome: 'Chromosome', context: 'ExecutionContext'):
        print(f"[Executor] Ejecutando estrategia: '{chromosome.description}'")
        loop = asyncio.get_running_loop()

        for gene in chromosome.genes:
            print(f"  -> Próximo gen en la secuencia: {gene.__class__.__name__}")
            try:
                # La clave está aquí: ejecutamos el método síncrono 'execute'
                # en un ejecutor de hilos separado para no bloquear el bucle de eventos.
                await loop.run_in_executor(
                    None,  # Usa el ejecutor de hilos por defecto
                    gene.execute,  # La función síncrona a ejecutar
                    context  # El argumento para la función
                )
            except Exception as e:
                print(f"[ERROR FATAL EN EXECUTOR] El gen '{gene.__class__.__name__}' falló.")
                import traceback
                traceback.print_exc()
                context.log_thought("FATAL_EXECUTOR_ERROR", f"Error en {gene.__class__.__name__}: {e}")
                break
class FindRelatedConceptsGene(Gene):
    """
    Encuentra todos los nodos directamente conectados a un concepto en el Knowledge Graph.
    """
    def __init__(self, concept_var: str, output_var: str, graph: KnowledgeGraph):
        self.concept_var = concept_var
        self.output_var = output_var
        self.graph = graph

    def execute(self, context: ExecutionContext):
        concepts = context.get(self.concept_var, [])
        if not concepts:
            context.set(self.output_var, [])
            return
            
        primary_concept = str(concepts[0]).lower()
        related_concepts = []

        # Usamos el método del KnowledgeGraph para obtener los vecinos
        if self.graph.has_node(primary_concept):
            related_concepts = self.graph.get_all_neighbors(primary_concept)

        print(f"  [Gene] FindRelated: Para '{primary_concept}', encontrados {len(related_concepts)} conceptos relacionados.")
        context.set(self.output_var, related_concepts)


class InferRelationshipGene(Gene):
    """
    Intenta encontrar un camino o una conexión común entre dos conceptos en el Knowledge Graph.
    """
    def __init__(self, concepts_var: str, output_var: str, graph: KnowledgeGraph):
        self.concepts_var = concepts_var
        self.output_var = output_var
        self.graph = graph

    def execute(self, context: ExecutionContext):
        concepts = context.get(self.concepts_var, [])
        if len(concepts) < 2:
            context.set(self.output_var, "Necesito al menos dos conceptos para inferir una relación.")
            return

        concept1 = str(concepts[0]).lower()
        concept2 = str(concepts[1]).lower()
        
        if not self.graph.has_node(concept1) or not self.graph.has_node(concept2):
            context.set(self.output_var, "No tengo suficiente información sobre uno o ambos conceptos.")
            return

        # 1. Intentar encontrar un camino directo
        try:
            path = self.graph.find_shortest_path(concept1, concept2)
            if path and 1 < len(path) < 5:
                relationship_text = " -> ".join(f"'{p}'" for p in path)
                result = f"He encontrado una conexión directa entre ellos: {relationship_text}."
                context.set(self.output_var, result)
                print(f"  [Gene] InferRelationship: Path encontrado: {result}")
                return
        except Exception as e:
            print(f"  [WARN] InferRelationship: Error al buscar camino en KG: {e}")
            pass

        # 2. Si no hay camino, buscar vecinos en común
        try:
            neighbors1 = set(self.graph.get_all_neighbors(concept1))
            neighbors2 = set(self.graph.get_all_neighbors(concept2))
            common_neighbors = neighbors1.intersection(neighbors2)

            if common_neighbors:
                result = f"No están conectados directamente, pero ambos se relacionan con los siguientes conceptos: {', '.join(f'{c}' for c in common_neighbors)}."
                context.set(self.output_var, result)
                print(f"  [Gene] InferRelationship: Vecinos comunes encontrados: {result}")
                return
        except Exception as e:
            print(f"  [WARN] InferRelationship: Error al buscar vecinos comunes en KG: {e}")
            pass
        
        context.set(self.output_var, "No he podido encontrar una relación clara entre ellos en mi base de conocimiento.")
        print(f"  [Gene] InferRelationship: No se encontró relación entre '{concept1}' y '{concept2}'.")

class GoalManager:
    """
    Gestiona un árbol jerárquico de objetivos. Prioriza y rastrea el estado de las
    ambiciones a corto, mediano y largo plazo de la IA.
    """
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.goals: Dict[str, Goal] = {}
        self.prometheus = prometheus_mind

    def add_goal(self, description: str, target: Any, priority: int = 10, parent_id: str | None = None):
        """Añade un nuevo objetivo, posiblemente como sub-objetivo de otro."""
        new_goal = Goal(description, target, priority=priority, parent_goal_id=parent_id)

        self.goals[new_goal.id] = new_goal
        
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(new_goal)
            
        print(f"[GOAL_MANAGER] Nuevo objetivo añadido: {new_goal}")
        return new_goal

    def get_active_goal(self) -> Goal | None:
        """Encuentra el objetivo activo de mayor prioridad."""
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        if not active_goals:
            return None
        return max(active_goals, key=lambda g: g.priority)

    def set_goal_status(self, goal_id: str, status: GoalStatus):
        """Cambia el estado de un objetivo (ej. para pausarlo o reactivarlo)."""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            print(f"[GOAL_MANAGER] Estado del objetivo {goal_id[:4]}... cambiado a {status.value}")

    def update_goals_status(self, current_observation: Dict):
        """Comprueba y actualiza el estado de los objetivos activos basado en la observación."""
        active_goal = self.get_active_goal()
        if not active_goal:
            return False
            
        if np.array_equal(current_observation.get("agent"), active_goal.target):
            self.set_goal_status(active_goal.id, GoalStatus.COMPLETED)
            self.prometheus.episodic_memory.record_event(event_type="GOAL_COMPLETED", details={"description": active_goal.description})
            print(f"[GOAL_MANAGER] ¡OBJETIVO PRINCIPAL COMPLETADO!: {active_goal.description}")
            return True
        return False

    def execute(self, context: ExecutionContext):
        if self.goal_manager.get_active_goal():
            return

        corner = random.choice([[0,0], [0, self.mind.env_size-1], [self.mind.env_size-1, 0], [self.mind.env_size-1, self.mind.env_size-1]])
        target_pos = np.array(corner)
        description = f"Explorar la esquina en {target_pos}"

        new_goal = self.goal_manager.add_goal(
            description=description,
            target=target_pos
        )
        
        context.set("self_generated_goal", new_goal.description)


class GridWorldEnv(gym.Env):
    """
    Un entorno simple de mundo en cuadrícula (GridWorld).
    El agente (Prometheus) debe navegar desde una posición inicial hasta una meta.
    """
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, size=10, render_mode='ansi'):
        super(GridWorldEnv, self).__init__()
        self.size = size
        self.render_mode = render_mode
        
        # El espacio de acciones: 0: Arriba, 1: Abajo, 2: Izquierda, 3: Derecha
        self.action_space = gym.spaces.Discrete(4)
        
        # El espacio de observación: la posición (x, y) del agente y de la meta
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "goal": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })
        
        self.agent_pos = None
        self.goal_pos = None

    def _get_obs(self):
        return {"agent": self.agent_pos, "goal": self.goal_pos}

    def _get_info(self):
        # La distancia de Manhattan es una buena métrica de progreso
        return {"distance": np.abs(self.agent_pos - self.goal_pos).sum()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Colocar al agente y la meta en posiciones aleatorias no superpuestas
        self.agent_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.goal_pos = self.agent_pos
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
            
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Mapeo de acción a cambio de dirección
        direction_map = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        move = direction_map.get(action)
        
        # Guardar la posición anterior para calcular la recompensa
        prev_distance = self._get_info()["distance"]
        
        # Actualizar la posición del agente, asegurándose de que no se salga de los límites
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        
        # Calcular si el episodio ha terminado
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        
        # --- Lógica de Recompensa ---
        if terminated:
            reward = 100.0  # Gran recompensa por alcanzar la meta
        elif self._get_info()["distance"] < prev_distance:
            reward = 5.0   # Pequeña recompensa por acercarse
        else:
            reward = -1.0  # Pequeño castigo por cada paso para fomentar la eficiencia
            
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            grid = np.full((self.size, self.size), "_", dtype=str)
            grid[tuple(self.agent_pos)] = "P" # Prometheus
            grid[tuple(self.goal_pos)] = "G"  # Goal
            return "\n".join(" ".join(row) for row in grid)



class EvolutionEngine:
    def __init__(self, arsenal: List[Gene], elite_size: int, mutation_config: Dict[str, float]):
        self.arsenal = arsenal
        self.elite_size = elite_size
        self.mutation_config = mutation_config
        
    def _mutate(self, chromosome: Chromosome):
        """
        Despachador de mutaciones. Aplica una o más mutaciones a un cromosoma
        basándose en las probabilidades configuradas.
        """
        mutation_type = random.choices(
            list(self.mutation_config.keys()),
            list(self.mutation_config.values()),
            k=1
        )[0]

        if mutation_type == 'swap':
            self._mutate_swap(chromosome)
        elif mutation_type == 'insertion':
            self._mutate_insertion(chromosome)
        elif mutation_type == 'deletion':
            self._mutate_deletion(chromosome)
        elif mutation_type == 'rearrangement':
            self._mutate_rearrangement(chromosome)
        elif mutation_type == 'parameter':
            self._mutate_parameter(chromosome)
        
        return chromosome

    def _mutate_swap(self, ch: Chromosome):
        """Reemplaza un gen aleatorio por uno nuevo del arsenal. (La mutación original)."""
        if not ch.genes or not self.arsenal: return
        point = random.randint(0, len(ch.genes) - 1)
        ch.genes[point] = random.choice(self.arsenal)
        ch.description += " (Mut:Swap)"

    def _mutate_insertion(self, ch: Chromosome):
        """Inserta un nuevo gen aleatorio del arsenal en una posición aleatoria."""
        if not self.arsenal: return
        point = random.randint(0, len(ch.genes))
        ch.genes.insert(point, random.choice(self.arsenal))
        ch.description += " (Mut:Ins)"

    def _mutate_deletion(self, ch: Chromosome):
        """Elimina un gen aleatorio, si el cromosoma tiene más de un gen."""
        if len(ch.genes) > 1:
            point = random.randint(0, len(ch.genes) - 1)
            ch.genes.pop(point)
            ch.description += " (Mut:Del)"
            
    def _mutate_rearrangement(self, ch: Chromosome):
        """Intercambia la posición de dos genes aleatorios."""
        if len(ch.genes) > 1:
            idx1, idx2 = random.sample(range(len(ch.genes)), 2)
            ch.genes[idx1], ch.genes[idx2] = ch.genes[idx2], ch.genes[idx1]
            ch.description += " (Mut:Rearr)"

    def _mutate_parameter(self, ch: Chromosome):
        """
        Intenta cambiar un parámetro numérico (int, float) dentro de un gen aleatorio.
        Esta es una mutación de "ajuste fino".
        """
        if not ch.genes: return
        
        gene_to_mutate = random.choice(ch.genes)
        attrs = vars(gene_to_mutate)
        
        tunable_params = [k for k, v in attrs.items() if isinstance(v, (int, float))]
        
        if not tunable_params: return

        param_name = random.choice(tunable_params)
        current_value = getattr(gene_to_mutate, param_name)

        if isinstance(current_value, int):
            change = random.choice([-1, 1])
            new_value = max(1, current_value + change)
        elif isinstance(current_value, float):
            change_factor = random.uniform(0.8, 1.2)
            new_value = current_value * change_factor
        
        setattr(gene_to_mutate, param_name, new_value)
        ch.description += f" (Mut:Param {param_name[:3]})"

    def _crossover_two_point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Intercambia el segmento central entre dos cromosomas."""
        if len(p1.genes) < 2 or len(p2.genes) < 2:
            return self._crossover_single_point(p1, p2)

        len1, len2 = len(p1.genes), len(p2.genes)
        size = min(len1, len2)
        
        pt1 = random.randint(0, size - 2)
        pt2 = random.randint(pt1 + 1, size - 1)
        
        c1_genes = p1.genes[:pt1] + p2.genes[pt1:pt2] + p1.genes[pt2:]
        c2_genes = p2.genes[:pt1] + p1.genes[pt1:pt2] + p2.genes[pt2:]
        
        return Chromosome(c1_genes, "Hijo de Cruce 2P"), Chromosome(c2_genes, "Hijo de Cruce 2P")

    def _crossover_uniform(self, p1: Chromosome, p2: Chromosome, swap_prob: float = 0.5) -> Tuple[Chromosome, Chromosome]:
        """Para cada posición de gen, decide aleatoriamente si se intercambian o no."""
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

    def _crossover_single_point(self, p1: Chromosome, p2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """El cruce original, ahora como un método separado."""
        len1, len2 = len(p1.genes), len(p2.genes)
        if len1 == 0 or len2 == 0: return copy.deepcopy(p1), copy.deepcopy(p2)

        point = random.randint(1, min(len1, len2) -1) if min(len1, len2) > 1 else 1
        c1_genes = p1.genes[:point] + p2.genes[point:]
        c2_genes = p2.genes[:point] + p1.genes[point:]
        return Chromosome(c1_genes, "Hijo de Cruce 1P"), Chromosome(c2_genes, "Hijo de Cruce 1P")


    def evolve(self, pop: List[Chromosome]) -> List[Chromosome]:
        """
        Ciclo de evolución principal, ahora usando un conjunto diverso de operadores.
        """
        pop.sort(key=lambda c: c.fitness, reverse=True)
        new_pop = pop[:self.elite_size]

        while len(new_pop) < len(pop):
            p1, p2 = self._tournament_selection(pop), self._tournament_selection(pop)
            if not p1 or not p2: continue

            crossover_method = random.choice([
                self._crossover_single_point, 
                self._crossover_two_point,
                self._crossover_uniform
            ])
            child1, child2 = crossover_method(p1, p2)

            new_pop.append(self._mutate(child1))
            if len(new_pop) < len(pop):
                new_pop.append(self._mutate(child2))
        
        return new_pop

    def _tournament_selection(self, pop: List[Chromosome], k: int = 3) -> Chromosome:
        """Selección por torneo para elegir a los padres."""
        if not pop: return None
        selection = random.choices(pop, k=k)
        return max(selection, key=lambda c: c.fitness)

class DecideActionGene(Gene):
    """
    Toma una decisión de acción basada en una jerarquía: Objetivo > Memoria > Exploración.
    """
    def __init__(self, output_var: str = "chosen_action"):
        self.output_var = output_var
        self.action_space = [0, 1, 2, 3] # 0:Arriba, 1:Abajo, 2:Izquierda, 3:Derecha
        self.action_map = {0: "ARRIBA", 1: "ABAJO", 2: "IZQUIERDA", 3: "DERECHA"}

    # REEMPLAZA EL MÉTODO execute() COMPLETO EN LA CLASE DecideActionGene

    def execute(self, context: ExecutionContext):
        active_goal = context.get("active_goal")
        agent_pos = context.get("agent_pos")
        action = None

        if active_goal and agent_pos is not None:
            # Convertimos a arrays de numpy para una fácil operación matemática
            target_pos = np.array(active_goal.target)
            agent_pos_arr = np.array(agent_pos)
            
            # Calculamos el vector de diferencia
            delta = target_pos - agent_pos_arr
            delta_y, delta_x = delta[0], delta[1]

            # Si ya estamos en la meta, no hacemos nada (acción None).
            # Esto permite que CuriosityGene se active en el siguiente ciclo.
            if delta_y == 0 and delta_x == 0:
                action = None
                print(f"  [Gene] DecideAction (Goal-Reached): Ya estoy en el objetivo {target_pos}.")
            
            # Priorizamos el movimiento en el eje con mayor distancia
            elif abs(delta_y) > abs(delta_x):
                action = 1 if delta_y > 0 else 0  # 1: Abajo, 0: Arriba
                print(f"  [Gene] DecideAction (Goal-Driven): Moviéndome en vertical ({self.action_map[action]}) para alcanzar {target_pos}.")
            
            # Si la distancia horizontal es mayor o igual, nos movemos horizontalmente
            else:
                action = 3 if delta_x > 0 else 2  # 3: Derecha, 2: Izquierda
                print(f"  [Gene] DecideAction (Goal-Driven): Moviéndome en horizontal ({self.action_map[action]}) para alcanzar {target_pos}.")
        
        # Si no hay objetivo, exploramos aleatoriamente
        if action is None and (not active_goal or (delta_y == 0 and delta_x == 0)):
             action = random.choice(self.action_space)
             print(f"  [Gene] DecideAction (Exploratory): Sin objetivo activo. Explorando hacia {self.action_map[action]}.")

        context.set(self.output_var, action)


class MetacognitionEngine:
    """
    Analiza el rendimiento de la IA en un lote de desafíos y evoluciona 
    los perfiles de fitness para mejorar el juicio en futuras evoluciones.
    """
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind

    def analyze_and_evolve_profiles(self, batch_results: List[Dict]):
        """
        Revisa los resultados de un lote y muta los perfiles de fitness correspondientes.
        """
        print("\n===== CICLO DE METACOGNICIÓN: EVOLUCIONANDO CRITERIOS DE JUICIO =====")
        
        # Agrupar los resultados por la intención de cada desafío
        results_by_intent = {}
        for res in batch_results:
            intent = res.get('intent')
            if intent:
                results_by_intent.setdefault(intent, []).append(res['score'])

        for intent, scores in results_by_intent.items():
            if not scores: continue
            
            avg_score = sum(scores) / len(scores)
            profile = self.mind.get_profile_for_intent(intent)
            
            # Actualizar el log de rendimiento en el SelfModel para el cálculo de confianza futuro
            self.mind.self_model.update_performance_log(intent, avg_score)

            # Meta-fitness: Si el rendimiento promedio es bajo, la tasa de aprendizaje (mutación) es alta.
            # Un score perfecto (1000) resulta en una tasa de mutación de 0.
            # Un score bajo resulta en una tasa de mutación alta para forzar un cambio.
            learning_rate = 0.5 * (1 - (avg_score / 1000.0))
            
            print(f"  Intención '{intent}': Rendimiento promedio={avg_score:.2f}. Tasa de mutación del perfil={learning_rate:.3f}")
            profile.mutate(learning_rate)
        
        print("====================== METACOGNICIÓN FINALIZADA ======================\n")

class WorldStream(Process):
    """
    Un proceso independiente que ejecuta el entorno simulado en un bucle constante.
    Se comunica con la mente principal a través de colas (Queues).
    """
    def __init__(self, environment: gym.Env):
        super().__init__()
        self.env = environment
        
        self.perception_queue = Queue()
        self.action_queue = Queue()
        
        self.stop_event = Event()

    def run(self):
        """El bucle de vida del mundo. Se ejecuta cuando llamamos a .start()"""
        print(f"[WORLD_STREAM pid={os.getpid()}] El mundo ha cobrado vida.")
        observation, info = self.env.reset()
        
        initial_percept = Percept(type="observation", data={"obs": observation, "info": info, "reward": 0, "terminated": False})
        self.perception_queue.put(initial_percept)

        while not self.stop_event.is_set():
            action = None
            if not self.action_queue.empty():
                action = self.action_queue.get()

            if action is not None:
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                percept_type = "termination" if terminated else "observation"
                percept = Percept(type=percept_type, data={"obs": observation, "info": info, "reward": reward, "terminated": terminated})
                self.perception_queue.put(percept)
                
                if terminated or truncated:
                    print("[WORLD_STREAM] Episodio terminado. Reiniciando entorno...")
                    observation, info = self.env.reset()
                    reset_percept = Percept(type="observation", data={"obs": observation, "info": info, "reward": 0, "terminated": False})
                    self.perception_queue.put(reset_percept)
            
            time.sleep(0.1)

    def stop(self):
        """Señal para terminar el proceso del mundo de forma segura."""
        self.stop_event.set()

class StrategicPlanner:
    """
    Decide dinámicamente qué genes usar para una consulta, reemplazando
    la necesidad de un bloque 'match/case' codificado.
    """
    def __init__(self, gene_descriptors: List[GeneDescriptor], full_arsenal: List[Gene], similarity_model, self_model: 'SelfModel'):
        self.gene_descriptors = gene_descriptors
        self.gene_map = {gene.__class__: gene for gene in full_arsenal}
        self.similarity_model = similarity_model
        self.self_model = self_model
        
        print("[PLANNER] Pre-calculando embeddings de capacidades del genoma...")
        self.descriptions_embeddings = self.similarity_model.encode(
            [desc.purpose_description for desc in self.gene_descriptors],
            convert_to_tensor=True
        )
        print(f"[PLANNER] {len(self.gene_descriptors)} capacidades analizadas y listas para la planificación.")

    

    def plan_seed_strategy(self, query: str, analysis: Dict) -> 'Chromosome':
        intent = analysis.get('intent', 'CONVERSACIONAL')
        print(f"  [Planner] Iniciando planificación dinámica para la intención '{intent}'...")

        initial_plan_descriptors = []
        available_variables = set(analysis.keys())
        
        resolve_context_descriptor = next((d for d in self.gene_descriptors if d.gene_class == ResolveContextGene), None)
        if resolve_context_descriptor and any(phrase in query.lower() for phrase in ["el tema anterior", "ese tema", "sobre eso", "dime más"]):
            initial_plan_descriptors.append(resolve_context_descriptor)
            if resolve_context_descriptor.output_variable:
                available_variables.add(resolve_context_descriptor.output_variable)
            print("    [Planner-Pre] Se añadió ResolveContextGene al inicio del plan.")

        relevant_descriptors = [
            desc for desc in self.gene_descriptors 
            if (intent in desc.relevant_intents or not desc.relevant_intents) and desc.gene_class != ResolveContextGene
        ]

        print(f"    [Planner-Filter] {len(relevant_descriptors)} genes relevantes encontrados para '{intent}'.")

        query_embedding = self.similarity_model.encode(query, convert_to_tensor=True)
        
        if not relevant_descriptors:
            candidate_descriptors = []
        else:
            relevant_embeddings = self.similarity_model.encode(
                [desc.purpose_description for desc in relevant_descriptors],
                convert_to_tensor=True
            )
            similarities = util.pytorch_cos_sim(query_embedding, relevant_embeddings)[0]
            top_k = min(5, len(relevant_descriptors))
            top_results = torch.topk(similarities, k=top_k)
            candidate_descriptors = [relevant_descriptors[i] for i in top_results.indices]
            
            # --- ORDENACIÓN POR PRIORIDAD (AQUÍ ESTÁ LA CORRECCIÓN CLAVE) ---
            candidate_descriptors.sort(key=lambda d: d.priority, reverse=True)
        
        print(f"    [Planner-Select] Genes candidatos por relevancia y prioridad: {[g.gene_class.__name__ for g in candidate_descriptors]}")

        ordered_plan = []
        candidates_to_sequence = candidate_descriptors.copy()
        
        for _ in range(len(candidates_to_sequence) + 1):
            gene_added_in_pass = False
            remaining_candidates = []
            for descriptor in candidates_to_sequence:
                if set(descriptor.input_variables).issubset(available_variables):
                    ordered_plan.append(descriptor)
                    if descriptor.output_variable:
                        available_variables.add(descriptor.output_variable)
                    gene_added_in_pass = True
                else:
                    remaining_candidates.append(descriptor)
            
            candidates_to_sequence = remaining_candidates
            if not gene_added_in_pass or not candidates_to_sequence:
                break
        
        if candidates_to_sequence:
            ordered_plan.extend(candidates_to_sequence)
        
        terminal_genes = [g for g in ordered_plan if g.is_terminal]
        plan_without_terminals = [g for g in ordered_plan if not g.is_terminal]
        
        if not terminal_genes:
            terminal_genes.append(next(d for d in self.gene_descriptors if d.gene_class == FormulateResponseGene))

        final_sequence_descriptors = initial_plan_descriptors + plan_without_terminals + terminal_genes
        
        final_gene_instances = [copy.deepcopy(self.gene_map[descriptor.gene_class]) for descriptor in final_sequence_descriptors]        
        plan_description = " -> ".join([g.__class__.__name__.replace('Gene', '') for g in final_gene_instances])
        
        print(f"    [Planner-Final] Plan decidido: {plan_description}")
        return Chromosome(final_gene_instances, f"Estrategia Dinámica: {plan_description}")



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

class PrometheusAGI:
    # EN: la clase PrometheusAGI
# REEMPLAZA el método __init__ con esta versión que incluye herramientas

    # Dentro de la clase PrometheusAGI en PROMETEUSV5.py

    def __init__(self, force_genesis: bool = False, **kwargs):
        """
        Constructor completo y corregido para la clase PrometheusAGI.
        Se ha ajustado el orden de inicialización para que los componentes
        se creen antes de ser utilizados por otros.
        """
        print("[MIND] Iniciando Prometheus con Núcleo Cognitivo y Uso de Herramientas...")
        self.force_genesis = force_genesis
        self.config = kwargs
        self.paths = PATHS
        self.conversational_history = []
        self.successful_strategies_archive: List['Chromosome'] = []
        self.intent_specialists: Dict[str, List['Chromosome']] = {}
        self.fitness_profiles_by_intent: Dict[str, 'FitnessProfile'] = {}
        self.gene_usage_counter = Counter()
        self.training_corpus = []

        # --- FASE 1: Componentes base ---
        print("[MIND] Cargando componentes base...")
        self.knowledge_graph = KnowledgeGraph()
        self.knowledge_graph.create_vector_index()
        self.episodic_memory = EpisodicMemory()
        self.similarity_model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        self._similarity_model = None # Se cargará de forma perezosa (lazy loading)

        try:
            self.lightweight_nlp = spacy.load("es_core_news_sm")
        except OSError:
            print("ADVERTENCIA: 'es_core_news_sm' no encontrado. Ejecuta: python -m spacy download es_core_news_sm")
            exit()
            
        self.gene_forge = GeneForge(self)
        self.self_model = SelfModel(self)
        self.goal_manager = GoalManager(self)
        self.failure_analyzer = FailureAnalysisEngine(self)

        # --- FASE 2: Genoma y Herramientas ---
        print("[MIND] Catalogando genes y herramientas...")
        self.tool_genes = {
            "WebSearchGene": WebSearchGene(),
            "CalculatorGene": CalculatorGene(),
            "ScientificSearchGene": ScientificSearchGene(),
        }
        self.tool_descriptors = self._create_tool_descriptors()
        self.full_arsenal = {
            "CognitiveCoreGene": CognitiveCoreGene(mind=self),
            "PerceiveEnvironmentGene": PerceiveEnvironmentGene(),
            "CheckGoalStatusGene": CheckGoalStatusGene(goal_manager=self.goal_manager),
            "CuriosityGene": CuriosityGene(mind=self),
            "DecideActionGene": DecideActionGene(),
            "WhyDidIFailGene": WhyDidIFailGene(failure_analyzer=self.failure_analyzer),
        }
        
        # --- FASE 3: Componentes Evolutivos (ORDEN CRÍTICO) ---

        # ===================================================================
        # AQUÍ ESTÁ LA CORRECCIÓN 
        #
        # A. Crear el motor de intenciones PRIMERO.
        #    Otros componentes, como la fábrica de genes, dependen de él.
        self.intent_engine = DynamicIntentEngine()

        # B. Crear el arsenal de genes "seguro" para el Dojo.
        self.dojo_picklable_arsenal = list(self.full_arsenal.values())

        # C. Crear la fábrica de genes y luego el arsenal completo.
        #    Esto ya no fallará porque `self.intent_engine` ya existe.
        gene_factory = self._create_gene_factory()
        complete_gene_arsenal = [constructor() for constructor in gene_factory.values()]

        # D. Crear el planificador estratégico, que necesita el arsenal completo.
        self.strategic_planner = StrategicPlanner(
            gene_descriptors=self._create_gene_descriptors(), 
            full_arsenal=complete_gene_arsenal,
            similarity_model=self.similarity_model, 
            self_model=self.self_model
        )
        
        # E. Crear el generador de población.
        self.population_generator = PopulationGenerator(
            mind=self, 
            intent_engine=self.intent_engine, 
            strategic_planner=self.strategic_planner
        )
        
        # F. Finalmente, crear el Dojo, que depende de todo lo anterior.
        self.dojo = Dojo(self)
        # ===================================================================

        # --- FASE 4: Carga de estado guardado ---
        self._load_mind_state()


    def _create_tool_descriptors(self) -> List[GeneDescriptor]:
        """
        Crea descriptores únicamente para los genes que actúan como herramientas.
        """
        return [
            GeneDescriptor(WebSearchGene, 'buscar en internet información general, hechos, noticias o sobre personas y lugares',
                           input_variables=['query'], output_variable='web_summary'),
            GeneDescriptor(CalculatorGene, 'resolver una operación matemática, calcular un resultado numérico o hacer cuentas',
                           input_variables=['query'], output_variable='calculation_result'),
            GeneDescriptor(ScientificSearchGene, 'buscar artículos científicos, papers, estudios o abstracts en bases de datos académicas',
                           input_variables=['query'], output_variable='scientific_summary'),
        ]

        
    @property
    def similarity_model(self):
        """Carga el SentenceTransformer solo la primera vez que se necesita."""
        if self._similarity_model is None:
            print(f"\n[MIND] Cargando modelo de similitud '{self.similarity_model_name}' (primera vez)...")
            self._similarity_model = SentenceTransformer(self.similarity_model_name)
        return self._similarity_model

    def __deepcopy__(self, memo):
        """
        Método de copia profunda personalizado para prevenir el error de 'pickle'.
        En lugar de copiar todo el objeto PrometheusAGI, que contiene elementos
        no serializables como modelos de spacy o conexiones a Neo4j, simplemente
        devolvemos una referencia al objeto original. Esto es seguro porque
        solo debe existir una instancia de la 'mente'.
        """
        memo[id(self)] = self
        return self
    
    def think_and_act(self, observation: Dict) -> 'Chromosome':
        """
        Un ciclo de pensamiento especializado para actuar en un entorno.
        Genera una estrategia simple y la ejecuta. En el futuro, podría evolucionar.
        """
        print(f"[DEBUG] 'think_and_act' ha recibido la observación: {observation}")

        ctx_vars = {"observation": observation}
        
        action_strategy = Chromosome([
              PerceiveEnvironmentGene(),
            CheckGoalStatusGene(goal_manager=self.goal_manager),
            CuriosityGene(self),
            # RecallEpisodeGene(memory=self.episodic_memory), # Removed because it's undefined
            DecideActionGene()
      ], "Instinto de Acción Curiosa y Dirigida por Metas")

        executor = ChromosomeExecutor()
        executor.execute(action_strategy, ctx_vars)
        return action_strategy
    
    # EN: la clase PrometheusAGI
# REEMPLAZA el método _create_gene_descriptors completo con esta versión priorizada:

    def _create_gene_descriptors(self) -> List[GeneDescriptor]:
        """
        Crea y devuelve una lista con la metainformación de CADA gen disponible,
        incluyendo sus intenciones y prioridades para una planificación eficiente.
        """
        return [
            # --- NIVEL 1: CONTEXTO Y ESTADO DEL USUARIO (MÁXIMA PRIORIDAD) ---
            GeneDescriptor(ResolveContextGene, 'resolver referencias contextuales en la conversación (ej. "el tema anterior")',
                        relevant_intents=[], # Universal
                        input_variables=['query'], output_variable='main_topic',
                        priority=100),

            GeneDescriptor(DetectSentimentGene, 'detectar el sentimiento del usuario',
                        relevant_intents=['CONVERSACIONAL', 'SALUDAR'],
                        input_variables=['query'], output_variable='user_sentiment',
                        priority=90),

            # --- NIVEL 2: ACCIONES PRINCIPALES (ALTA PRIORIDAD) ---
            GeneDescriptor(GetNodeDefinitionGene, 'definir o explicar un concepto usando el conocimiento interno',
                        relevant_intents=['DEFINIR', 'RELACIONAR'],
                        input_variables=['main_topic'], output_variable='definition_text',
                        priority=50),

            GeneDescriptor(CalculatorGene, 'resolver o calcular una operación matemática',
                        relevant_intents=['CALCULAR'],
                        input_variables=['query'], output_variable='calculation_result',
                        priority=50),

            GeneDescriptor(DeepReasoningGene, 'encontrar la relación o conexión entre dos ideas',
                        relevant_intents=['RELACIONAR'],
                        input_variables=['query'], output_variable='reasoning_result',
                        priority=50),
            
            GeneDescriptor(RecallLastResponseGene, 'recordar, repetir o rememorar la última respuesta en la conversación actual',
                        relevant_intents=['RECORDAR'],
                        output_variable='recalled_response',
                        priority=50),
            
            # --- NIVEL 3: ACCIONES EN RED (LIGERAMENTE MENOS PRIORITARIAS POR SER LENTAS) ---
            GeneDescriptor(WebSearchGene, 'buscar, investigar o encontrar información en internet',
                        relevant_intents=['BUSCAR_WEB', 'RELACIONAR', 'DEFINIR'],
                        input_variables=['main_topic'], output_variable='web_summary',
                        priority=45),

            GeneDescriptor(ScientificSearchGene, 'buscar artículos y papers en bases de datos científicas como PubMed',
                        relevant_intents=['BUSCAR_CIENCIA'],
                        input_variables=['main_topic_en'], # Depende de la traducción
                        output_variable='scientific_summary',
                        priority=45),

            # --- NIVEL 4: PRE-PROCESAMIENTO (PRIORIDAD MEDIA) ---
            # (El planner ya lo antepone por la dependencia, pero la prioridad ayuda a clarificar)
            GeneDescriptor(LinguisticAnalysisGene, 'analizar la gramática y sintaxis de una frase',
                        relevant_intents=['ANALIZAR_GRAMATICA'],
                        input_variables=['query'], output_variable='linguistic_analysis',
                        priority=30),
            

            # --- NIVEL 5: METACOGNICIÓN Y POST-PROCESAMIENTO (PRIORIDAD POR DEFECTO) ---
            GeneDescriptor(LearnFromTextGene, 'aprender o asimilar información de un resumen web a la memoria',
                        relevant_intents=['BUSCAR_WEB'],
                        input_variables=['web_summary', 'main_topic'],
                        priority=10),


            GeneDescriptor(ExplainPlanGene, 'explicar mi plan o mis objetivos actuales',
                        relevant_intents=['EXPLORAR', 'CONVERSACIONAL'],
                        output_variable='plan_explanation',
                        priority=10),

            GeneDescriptor(PatternFinderGene, 'analizar mis propias estrategias y buscar patrones',
                        relevant_intents=['ANALIZAR_ESTRATEGIAS'],
                        input_variables=['intent'], output_variable='patterns',
                        priority=10),

            GeneDescriptor(AnalyzeGeneSyntaxGene, 'analizar el código fuente de una de mis capacidades',
                        relevant_intents=['ANALIZAR_CODIGO'],
                        input_variables=['main_topic'], output_variable='syntax_analysis_report',
                        priority=10),

            GeneDescriptor(EvolveStrategyGene, 'pensar profundamente o evolucionar una estrategia compleja',
                        relevant_intents=['BUSCAR_WEB', 'RELACIONAR', 'ANALIZAR_ESTRATEGIAS'],
                        input_variables=['query', 'intent'],
                        priority=10),

            # --- NIVEL 6: ACCIONES DE FALLBACK (BAJA PRIORIDAD) ---
            GeneDescriptor(InferDefinitionFromNeighborsGene, 'si no existe una definición directa, intentar inferirla a partir de conceptos vecinos y relacionados en la memoria',
                        relevant_intents=['DEFINIR'],
                        input_variables=['main_topic'], output_variable='definition_text',
                        priority=5),

            # --- NIVEL 7: RESPUESTA FINAL (MÍNIMA PRIORIDAD, SON TERMINALES) ---
            GeneDescriptor(DynamicConversationalGene, 'responder de forma empática o conversacional',
                        relevant_intents=['CONVERSACIONAL', 'SALUDAR'],
                        input_variables=['user_sentiment'], is_terminal=True,
                        priority=1),

            GeneDescriptor(FormulateResponseGene, 'formular una respuesta final al usuario a partir de los datos recopilados',
                        relevant_intents=[], # Universal
                        is_terminal=True,
                        priority=0), # La más baja, es el último recurso

            GeneDescriptor(AnalizadorGramaticalGene, 'analizar la gramática o sintaxis de una frase',
                        relevant_intents=['ANALIZAR_GRAMATICA'],
                        input_variables=['query'], is_terminal=True,
                        priority=1),
        ]
    
    # EN: PrometheusAGI
    # REEMPLAZA el método _create_gene_factory entero
        # EN: PrometheusAGI
    def _create_gene_factory(self) -> Dict[str, callable]:
        """
        Crea un diccionario de constructores de genes (usando lambdas)
        con sus dependencias correctamente inyectadas.
        """
        # ===================================================================
        # AQUÍ ESTÁ LA CORRECCIÓN
        # 1. Definimos el mapa de genes ANTES de que se use en el lambda.
        #    Este mapa contiene las CLASES de los genes, no las instancias.
        gene_map_for_analysis = {
            'GetNodeDefinitionGene': GetNodeDefinitionGene,
            'WebSearchGene': WebSearchGene,
            'ScientificSearchGene': ScientificSearchGene,
            'CalculatorGene': CalculatorGene,
            'DeepReasoningGene': DeepReasoningGene,
            'AnalizadorGramaticalGene': AnalizadorGramaticalGene,
            'FormulateResponseGene': FormulateResponseGene,
            'CognitiveCoreGene': CognitiveCoreGene,
            'PerceiveEnvironmentGene': PerceiveEnvironmentGene,
            'CheckGoalStatusGene': CheckGoalStatusGene,
            'CuriosityGene': CuriosityGene,
            'DecideActionGene': DecideActionGene,
            'WhyDidIFailGene': WhyDidIFailGene,
            'EvolveStrategyGene': EvolveStrategyGene,
            # Añade cualquier otra clase de gen que quieras que sea analizable aquí
        }
        # ===================================================================

        return {
            # Genes que dependen de 'mind'
            'RecallLastResponseGene': lambda: RecallLastResponseGene(mind=self),
            'ResolveContextGene': lambda: ResolveContextGene(mind=self),
            'CuriosityGene': lambda: CuriosityGene(mind=self),
            'PatternFinderGene': lambda: PatternFinderGene(mind=self, archive_to_analyze="specialists", intent_var="intent", output_var="patterns"),
            'EvolveStrategyGene': lambda: EvolveStrategyGene(mind=self, generations=5, population_size=10),
            'WhyDidIFailGene': lambda: WhyDidIFailGene(failure_analyzer=self.failure_analyzer),

            # Genes que dependen de 'graph' y 'nlp'
            'DeepReasoningGene': lambda: DeepReasoningGene(graph=self.knowledge_graph, nlp_processor=self.intent_engine.nlp),
            'LinguisticAnalysisGene': lambda: LinguisticAnalysisGene(graph=self.knowledge_graph, nlp_processor=self.intent_engine.nlp),
            'InferDefinitionFromNeighborsGene': lambda: InferDefinitionFromNeighborsGene(graph=self.knowledge_graph, nlp_processor=self.intent_engine.nlp),
            
            # Genes que dependen de 'graph'
            'GetNodeDefinitionGene': lambda: GetNodeDefinitionGene(graph=self.knowledge_graph, entity_var="entities"),

            # Genes que dependen de 'nlp'
            'AnalizadorGramaticalGene': lambda: AnalizadorGramaticalGene(nlp_processor=self.intent_engine.nlp),

            # Genes que dependen de otros objetos
            'CheckGoalStatusGene': lambda: CheckGoalStatusGene(goal_manager=self.goal_manager),
            'ExplainPlanGene': lambda: ExplainPlanGene(goal_manager=self.goal_manager),
            
            # --- CORRECCIÓN APLICADA AQUÍ ---
            # Ahora el lambda puede encontrar 'gene_map_for_analysis' porque está en su mismo scope.
            'AnalyzeGeneSyntaxGene': lambda: AnalyzeGeneSyntaxGene(gene_name_var="main_topic", output_var="syntax_analysis_report", gene_map=gene_map_for_analysis),
            'DynamicConversationalGene': lambda: DynamicConversationalGene(),
            # Genes simples (sin dependencias complejas)
            'WebSearchGene': lambda: WebSearchGene(),
            'ScientificSearchGene': lambda: ScientificSearchGene(),
            'CalculatorGene': lambda: CalculatorGene(),
            'DetectSentimentGene': lambda: DetectSentimentGene(),
            'FormulateResponseGene': lambda: FormulateResponseGene(),
            'LearnFromTextGene': lambda: LearnFromTextGene(text_var="web_summary", graph=self.knowledge_graph, nlp_processor=self.intent_engine.nlp, similarity_model=self.similarity_model),
            'LearnFromScienceGene': lambda: LearnFromTextGene(text_var="scientific_summary", graph=self.knowledge_graph, nlp_processor=self.intent_engine.nlp, similarity_model=self.similarity_model),
        }

    def _load_mind_state(self):
        """
        Carga los componentes de la mente desde archivos pickle.
        El Knowledge Graph se ignora aquí, ya que está "vivo" en la base de datos Neo4j.
        """
        print("[MIND] Comprobando estado mental guardado (archivos pickle)...")
        os.makedirs(MIND_STATE_DIR, exist_ok=True)

        pickle_files_to_load = {
            "episodic_memory": "episodic_memory",
            "general_archive": "successful_strategies_archive",
            "specialists_archive": "intent_specialists",
            "fitness_profiles": "fitness_profiles_by_intent",
            "self_model_performance_log": ("self_model", "performance_log"), # Tupla para atributo anidado
            "gene_usage": "gene_usage_counter"
        }

        for path_key, attr_info in pickle_files_to_load.items():
            file_path = self.paths.get(path_key)
            if isinstance(attr_info, tuple): # Manejar atributos anidados (self_model.performance_log)
                attr_name = attr_info[0]
                sub_attr_name = attr_info[1]
                obj_to_set = getattr(self, attr_name)
            else: # Atributos directos
                attr_name = attr_info
                obj_to_set = self


            if not file_path: continue

            if os.path.exists(file_path) and not self.force_genesis:
                try:
                    with open(file_path, "rb") as f:
                        data = pickle.load(f)
                        if isinstance(attr_info, tuple):
                            setattr(obj_to_set, sub_attr_name, data)
                        else:
                            setattr(self, attr_name, data)
                    print(f"[MIND] '{path_key}' cargado desde '{file_path}'.")
                except Exception as e:
                    print(f"[WARN] No se pudo cargar '{path_key}' desde '{file_path}': {e}")

        try:
            corpus_path = self.paths['dojo_dataset']
            self.training_corpus = []
            with open(corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Carga cada línea como un objeto JSON independiente
                    self.training_corpus.append(json.loads(line))
            print(f"[MIND] Curriculum del Dojo cargado con {len(self.training_corpus)} lecciones desde '{corpus_path}'.")
        except FileNotFoundError:
             print(f"[WARN] No se encontró el archivo de dataset '{corpus_path}'. El modo Maratón podría no funcionar.")
        except Exception as e:
            print(f"[WARN] No se pudo cargar el dataset del Dojo: {e}")

    def _save_mind_state(self):
        print("[MIND] Guardando estado mental (archivos pickle)...")
        
        save_map = {
            "episodic_memory": self.episodic_memory,
            "general_archive": self.successful_strategies_archive,
            "specialists_archive": self.intent_specialists,
            "fitness_profiles": self.fitness_profiles_by_intent,
            "self_model_performance_log": self.self_model.performance_log,
            "gene_usage": self.gene_usage_counter
        }

        for path_key, data_obj in save_map.items():
            file_path = self.paths.get(path_key)
            if not file_path: continue
            
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(data_obj, f)
            except Exception as e:
                print(f"[ERROR] No se pudo guardar el archivo '{file_path}': {e}")
                
        print("[MIND] Componentes de memoria guardados en archivos pickle.")
        print("[MIND] El Knowledge Graph ya está sincronizado con Neo4j.")

    def shutdown(self):
        """
        Realiza un apagado seguro de Prometheus, cerrando las conexiones a las
        bases de datos y guardando cualquier estado pendiente.
        """
        print("\n[MIND] Iniciando secuencia de apagado de Prometheus...")
        self.knowledge_graph.close()
        self._save_mind_state()
        print("[MIND] Apagado completado.")

    def get_profile_for_intent(self, intent: str) -> 'FitnessProfile':
        """Obtiene el perfil de fitness para una intención, creándolo si no existe."""
        if intent not in self.fitness_profiles_by_intent:
            print(f"[MIND] Creando nuevo perfil de fitness para la intención: '{intent}'")
            self.fitness_profiles_by_intent[intent] = FitnessProfile(intent)
        return self.fitness_profiles_by_intent[intent]


    def update_specialist(self, intent: str, chromosome: 'Chromosome'):
        """Actualiza la lista de especialistas para una intención con un nuevo cromosoma exitoso."""
        if not intent: return
        print(f"[MIND] Considerando nuevo especialista para la intención '{intent}' (Fitness: {chromosome.fitness:.2f})")
        
        specialists = self.intent_specialists.get(intent, [])
        specialists.append(chromosome)
        specialists.sort(key=lambda c: c.fitness, reverse=True)
        self.intent_specialists[intent] = specialists[:self.MAX_SPECIALISTS_PER_INTENT]


    def get_specialists_for_intent(self, intent: str) -> List['Chromosome']:
        """Devuelve la lista de cromosomas de élite para una intención dada."""
        return self.intent_specialists.get(intent, [])
        
    def archive_strategy(self, chromosome: 'Chromosome'):
        """Añade un cromosoma exitoso al archivo general, manteniendo el orden por fitness."""
        if not isinstance(chromosome, Chromosome): return
        self.successful_strategies_archive.append(chromosome)
        self.successful_strategies_archive.sort(key=lambda c: c.fitness, reverse=True)
        self.successful_strategies_archive = self.successful_strategies_archive[:100]


    def record_gene_usage(self, chromosome: 'Chromosome'):
        """Registra qué genes se han usado en un cromosoma para el cálculo de curiosidad."""
        for gene in chromosome.genes:
            self.gene_usage_counter[gene.__class__.__name__] += 1

    async def think(self, query: str) -> str:
        """
        Procesa una consulta del usuario de forma unificada y generativa.
        """
        print(f"\n[MIND] Procesando cognitivamente: '{query}'...")

        # El único "plan" es ejecutar el núcleo cognitivo.
        # Ya no se analiza la intención, se pasa directamente al CognitiveCoreGene.
        cognitive_strategy = Chromosome(
            genes=[self.full_arsenal["CognitiveCoreGene"]], # Accedemos al gen desde el diccionario
            description="Estrategia Cognitiva Central"
        )

        # Preparamos el contexto inicial simple
        ctx_vars = {'query': query}
        
        # Ejecutamos la estrategia
        final_context = ExecutionContext(initial_vars=ctx_vars)
        await ChromosomeExecutor.execute_async(cognitive_strategy, final_context)
        
        final_response = final_context.get_final_response()
        
        if not final_response:
            final_response = "He procesado tu solicitud, pero no he podido formular una respuesta."
            
        # Guardamos en el historial de conversación
        self.conversational_history.append({"user": query, "prometheus": final_response})
        self.conversational_history = self.conversational_history[-5:]

        return final_response


    def attempt(self, instruction: str, ideal_response: str) -> tuple[str, float, 'Chromosome']:
        """
        Intento rápido para resolver un desafío, usado principalmente para la evaluación inicial en el Dojo.
        """
        strategies_to_try = self.successful_strategies_archive[:10]
        if not strategies_to_try:
            return "Sin estrategias.", 0.0, Chromosome([], "Fallback Vacío")

        best_chromosome = strategies_to_try[0]
        executor = ChromosomeExecutor()
        executor.execute(best_chromosome, {'query': instruction})
        actual_response = best_chromosome.final_context.get_final_response()

        try:
            emb1 = self.similarity_model.encode(actual_response, convert_to_tensor=True)
            emb2 = self.similarity_model.encode(ideal_response, convert_to_tensor=True)
            score = util.pytorch_cos_sim(emb1, emb2).item() * 1000
        except Exception:
            score = 0.0
            
        return actual_response, score, best_chromosome

class FailureAnalysisEngine:
    """
    Realiza un análisis post-mortem de los fracasos para determinar la causa raíz.
    """
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind

    def analyze_last_failure(self) -> str:
        """
        Analiza la última secuencia de fallo y devuelve un diagnóstico.
        """
        failure_context = self.mind.episodic_memory.get_last_failed_goal_context()
        if not failure_context:
            return "No encuentro registros de fallos recientes para analizar."

        # El evento de fallo es el último en el contexto
        failed_goal_details = failure_context[-1]["details"]
        
        # Analizar los eventos de acción que llevaron al fallo
        action_events = [event for event in failure_context if event["type"] == "ACTION_TAKEN"]
        
        if not action_events:
            return f"Fallé en el objetivo '{failed_goal_details['description']}' porque no tomé ninguna acción."

        # --- Lógica del Diagnóstico Diferencial ---
        # Analicemos la última acción tomada
        last_action_details = action_events[-1]["details"]
        
        fitness_profile = last_action_details["fitness_profile_used"]
        raw_scores = last_action_details["raw_scores"]
        
        # Hipótesis 1: ¿Fueron mis valores (perfil) el problema?
        # Ejemplo: Si el objetivo era llegar a la meta (alta similitud requerida),
        # pero mi perfil valoraba más la curiosidad.
        if "similarity" in raw_scores and "curiosity" in fitness_profile.weights:
            if fitness_profile.weights["curiosity"] > fitness_profile.weights["similarity"] * 20:
                # La IA se distrajo por ser "demasiado curiosa"
                diagnosis = (f"Fallé en el objetivo '{failed_goal_details['description']}'. "
                             "Diagnóstico: **Prioridades equivocadas**. "
                             f"Mi perfil de fitness actual '{fitness_profile.intent_name}' valora demasiado la curiosidad en detrimento de la precisión. "
                             "Se necesita una mutación en mis criterios de juicio.")
                # En una implementación completa, aquí se llamaría a profile.mutate() con un learning_rate alto.
                return diagnosis

        # Hipótesis 2: ¿Fue mi plan (cromosoma) el problema?
        # Ejemplo: Si mis valores eran correctos (valoraba la similitud), pero la acción que tomé
        # resultó en una puntuación de similitud baja.
        if "similarity" in raw_scores and raw_scores["similarity"] < 200: # Umbral de acción "mala"
            diagnosis = (f"Fallé en el objetivo '{failed_goal_details['description']}'. "
                         "Diagnóstico: **Estrategia defectuosa**. "
                         "Mis prioridades eran correctas, pero el plan de acción que ejecuté fue ineficaz y no me acercó al objetivo. "
                         "Necesito evolucionar mejores estrategias para esta situación.")
            return diagnosis
            
        return f"Fallé en el objetivo '{failed_goal_details['description']}', pero la causa raíz no es inmediatamente obvia. Requiere más análisis."



class SelfModel:
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.mind = prometheus_mind
        self.capability_map: Dict[str, Gene] = {}
        
        self.current_state: Dict[str, Any] = {
            "intent": None,
            "query": None,
            "goal": "En espera de instrucciones.",
            "confidence": 1.0
        }
        
        self.performance_log: Dict[str, List[float]] = {}

    def _initialize_capabilities(self) -> Dict[str, Gene]:
        """Crea un mapa de las capacidades a partir del arsenal de genes."""
        capability_map = {}
        for gene in self.mind.full_arsenal:
            capability_name = gene.__class__.__name__.replace("Gene", "")
            capability_map[capability_name] = gene
        return capability_map

    def register_new_capability(self, new_gene: Gene):
        """Método llamado por GeneForge para registrar un nuevo gen."""
        capability_name = new_gene.__class__.__name__.replace("Gene", "")
        self.capability_map[capability_name] = new_gene
        print(f"[SELF_MODEL] Nueva capacidad registrada: '{capability_name}'")

    def update_current_state(self, intent: str, query: str):
        """Actualiza el foco de atención de la IA."""
        self.current_state['intent'] = intent
        self.current_state['query'] = query
        self.current_state['goal'] = f"Responder a la consulta '{query[:30]}...' con la intención de '{intent}'."
        
        past_scores = self.performance_log.get(intent, [500])
        avg_score = sum(past_scores) / len(past_scores)
        self.current_state['confidence'] = avg_score / 1000.0
        
        print(f"[SELF_MODEL] Estado actualizado. Meta: {self.current_state['goal']}. Confianza: {self.current_state['confidence']:.2%}")
        
    def update_performance_log(self, intent: str, score: float):
        """Añade una nueva puntuación de rendimiento para una intención."""
        self.performance_log.setdefault(intent, []).append(score)
        self.performance_log[intent] = self.performance_log[intent][-20:]


class SelfReflectionGene(Gene):
    """
    Permite a la IA consultar su propio SelfModel para responder preguntas sobre sí misma.
    """
    def __init__(self, query_type: str, output_var: str, self_model: 'SelfModel'):
        self.query_type = query_type
        self.output_var = output_var
        self.self_model = self_model

    def execute(self, context: ExecutionContext):
        report = "No he podido reflexionar sobre mi estado."
        
        match self.query_type:
            case "list_capabilities":
                capabilities = list(self.self_model.capability_map.keys())
                report = f"Actualmente poseo las siguientes capacidades básicas: {', '.join(capabilities)}."
            
            case "get_status":
                state = self.self_model.current_state
                report = f"Mi estado actual es: \n- Objetivo: {state.get('goal')}\n- Confianza para esta tarea: {state.get('confidence', 0):.2%}"
            
            case "check_confidence":
                state = self.self_model.current_state
                report = f"Basado en mi rendimiento pasado, mi nivel de confianza para abordar la intención '{state.get('intent')}' es de {state.get('confidence', 0):.2%}."
        
        context.set(self.output_var, report)

# REEMPLAZA la función init_worker con esta versión completa
def init_worker(model_name: str, spacy_model_name: str, neo4j_uri: str, neo4j_user: str, neo4j_pass: str):
    """
    Inicializa CADA proceso del pool con todos los recursos necesarios.
    """
    global worker_model, worker_nlp, worker_kg
    print(f"[Worker-Init pid={os.getpid()}] Iniciando worker completo...")
    
    # Cargar modelo de similitud
    worker_model = SentenceTransformer(model_name)
    
    # Cargar modelo de lenguaje
    worker_nlp = spacy.load(spacy_model_name)
    
    # Crear conexión a la base de datos
    worker_kg = KnowledgeGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    
    print(f"[Worker-Init pid={os.getpid()}] Worker listo.")

# REEMPLAZA la función _worker_evaluate_chromosome_base completa con esta:

def _worker_evaluate_chromosome_base(args: Tuple):
    """
    El trabajador ahora recibe un diccionario de contexto completo (ctx_vars).
    """
    # Desempaquetamos los argumentos, incluyendo el nuevo ctx_vars
    chromosome, ctx_vars, ideal_response, gene_usage_counter = args

    executor = ChromosomeExecutor()
    calculator = FitnessCalculator(sim_model=worker_model)

    # Pasamos el diccionario de contexto completo al ejecutor
    executor.execute(chromosome, ctx_vars=ctx_vars)
    
    raw_scores = calculator.calculate_metrics(chromosome, ideal_response, gene_usage_counter)
    return chromosome, raw_scores
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
        num_batches = 500
        
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

def setup_environment():
    """
    Crea un entorno de prueba si no existe, ahora generando un archivo .jsonl.
    """
    print("--- Configurando entorno de prueba ---")
    os.makedirs(MIND_STATE_DIR, exist_ok=True)
    
    dojo_dataset_path = PATHS["dojo_dataset"]
    
    if not os.path.exists(dojo_dataset_path):
        print(f"Creando fichero de ejemplo '{dojo_dataset_path}'...")
        dummy_dataset = [
            {"INSTRUCTION": "¿Qué es la gravedad?", "RESPONSE": "La gravedad es la fuerza de atracción entre objetos con masa."},
            {"INSTRUCTION": "Define el concepto de 'democracia'.", "RESPONSE": "La democracia es un sistema de gobierno donde el poder reside en el pueblo."},
            {"INSTRUCTION": "Investiga sobre 'el futuro de la inteligencia artificial'", "RESPONSE": "La inteligencia artificial avanza hacia modelos más generales, éticos y eficientes, impactando la medicina, la ciencia y la vida diaria."},
            {"INSTRUCTION": "¿Qué relación hay entre 'ADN' y 'genética'?", "RESPONSE": "El ADN es la molécula que contiene la información genética, y la genética es la ciencia que estudia esta información y herencia."},
            {"INSTRUCTION": "Hola, ¿cómo estás?", "RESPONSE": "¡Hola! Estoy listo para ayudarte. ¿Qué necesitas?"}
        ]
        
        # --- SECCIÓN MODIFICADA PARA ESCRIBIR .jsonl ---
        with open(dojo_dataset_path, 'w', encoding='utf-8') as f:
            for entry in dummy_dataset:
                # Escribe cada diccionario como una línea JSON separada
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print("\n[IMPORTANTE] Para la funcionalidad de búsqueda web:")
    print("1. Asegúrate de tener instalada la librería: pip install google-api-python-client")
    print("2. Configura tus variables de entorno antes de ejecutar.")
    print("-" * 35)


def run_training_dojo(prometheus: "PrometheusAGI", steps: int):
    print("\n" + "="*60 + "\nINICIANDO MODO 'DOJO' DE ENTRENAMIENTO...\n" + "="*60)
    if not prometheus.training_corpus: print("\n[ERROR] No hay corpus de entrenamiento cargado."); return
    from itertools import cycle, islice
    for i, training_example in enumerate(islice(cycle(prometheus.training_corpus), steps)):
        print(f"\n--- DESAFÍO DEL DOJO {i + 1}/{steps} ---")
        # Aquí, 'think' ya no depende del LLM para la respuesta final.
        prometheus.think(query=training_example["INSTRUCTION"]) 
        if (i + 1) % 50 == 0: prometheus.save_state()
    print("\n" + "="*60 + "\nSESIÓN EN EL DOJO FINALIZADA.\n" + "="*60)
    prometheus.save_state()


CORPUS_FILENAME = "corpus_de_conocimiento.txt"
NUM_ARTICULOS_A_PROCESAR = 5000


def descargar_y_preparar_corpus():
    """
    Comprueba si el corpus local existe. Si no, lo descarga desde
    Hugging Face (Wikipedia en español) y lo guarda localmente.
    """
    if os.path.exists(CORPUS_FILENAME):
        return

    print("El corpus no existe localmente. Iniciando descarga desde Hugging Face...")
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.es", streaming=True, split="train")
        dataset_subset = dataset.take(NUM_ARTICULOS_A_PROCESAR)
        print(f"Descarga iniciada. Se procesarán {NUM_ARTICULOS_A_PROCESAR} artículos de Wikipedia...")

        with open(CORPUS_FILENAME, 'w', encoding='utf-8') as f:
            for i, article in enumerate(dataset_subset):
                text = article['text']
                text = re.sub(r'==.*?==', '', text)
                text = re.sub(r'\n+', '\n\n', text).strip()
                f.write(f"TEMA: {article['title']}\n")
                f.write(text)
                f.write("\n\n---\n\n")
                if (i + 1) % 100 == 0:
                    print(f"  ... {i + 1}/{NUM_ARTICULOS_A_PROCESAR} artículos procesados y guardados.")
        print(f"¡Éxito! El corpus ha sido creado en '{CORPUS_FILENAME}'.")
    except Exception as e:
        print(f"[ERROR FATAL] No se pudo descargar o procesar el corpus: {e}")
        exit()
async def run_interactive_mode_async(prometheus: "PrometheusAGI"):
    """Versión asíncrona del modo interactivo."""
    print("\n" + "="*60 + "\nINICIANDO MODO INTERACTIVO (Async)...\n" + "="*60)
    while True:
        try:
            user_input = await asyncio.to_thread(input, "\nTU ORDEN > ")
            user_input = user_input.strip()
            if user_input.lower() in ['exit', 'quit', 'salir']:
                break
            # Esperamos a que el proceso de pensamiento asíncrono termine
            ai_response = await prometheus.think(user_input)
            print("\n" + "-"*45 + f"\nPROMETEO RESPONDE: {ai_response}\n" + "-"*45)
        except KeyboardInterrupt:
            break
    prometheus._save_mind_state()

def run_huggingface_code_pretraining(dataset_name, split='train', language_filter='Python', batch_size=50):
    """
    Descarga y procesa en modo streaming un dataset de código desde Hugging Face,
    lo analiza con AST y lo inyecta en el Knowledge Graph.
    """
    print("="*60)
    print("INICIANDO PRE-ENTRENAMIENTO DESDE HUGGING FACE")
    print(f"Dataset: {dataset_name} | Filtro de Lenguaje: {language_filter}")
    print("="*60)

    try:
        prometheus = PrometheusAGI(force_genesis=True)
        
        # 1. Cargar el dataset en modo streaming para no consumir memoria RAM
        print("Conectando al stream del dataset en Hugging Face...")
        # NOTA: Para the-stack-v2, puede que necesites un token de acceso: use_auth_token=True
        dataset_stream = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        
        # 2. Filtrar por lenguaje si el dataset lo soporta
        # La clave 'lang' es común en datasets como the-stack, ajústala si es necesario
        if language_filter:
            dataset_stream = dataset_stream.filter(lambda example: example.get('lang') == language_filter)
            print(f"Stream filtrado. Procesando solo código en '{language_filter}'.")

        nodes_to_add = []
        edges_to_add = []
        processed_files = 0
        start_time = time.time()

        # 3. Iterar sobre el stream de datos
        for item in dataset_stream:
            code_content = item.get('content')
            # Creamos un nombre de módulo a partir de la ruta en el dataset, si existe
            module_name = item.get('path', f"hf_module_{processed_files}")

            if not code_content:
                continue

            try:
                tree = ast.parse(code_content, filename=module_name)
                
                # Reutilizamos nuestro analizador de código AST
                visitor = CodeVisitor(module_name)
                visitor.visit(tree)
                
                nodes_to_add.extend(visitor.nodes)
                edges_to_add.extend(visitor.edges)
                
                processed_files += 1
            except SyntaxError:
                # Es normal encontrar archivos con errores de sintaxis en grandes datasets
                continue
            except Exception as e:
                print(f"[ERROR] Error inesperado procesando {module_name}: {e}")

            # Inyectar en Neo4j por lotes
            if processed_files > 0 and processed_files % batch_size == 0:
                print(f"  ... Procesados {processed_files} archivos desde el stream. Inyectando lote en Neo4j...")
                prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)
                nodes_to_add, edges_to_add = [], [] # Limpiar
                gc.collect()

        # Inyectar el último lote
        if nodes_to_add or edges_to_add:
            print("  ... Inyectando último lote en Neo4j.")
            prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)

        end_time = time.time()
        print("\n--- Pre-entrenamiento desde Hugging Face completado. ---")
        prometheus.shutdown()
        print(f"Se analizaron {processed_files} archivos en {end_time - start_time:.2f} segundos.")
        print(f"El Knowledge Graph ahora contiene conocimiento del dataset '{dataset_name}'.")
        print("="*60)

    except Exception as e:
        print(f"[ERROR FATAL] Ocurrió un error durante el pre-entrenamiento desde Hugging Face:")
        import traceback
        traceback.print_exc()

def stream_code_files(root_dir, extension=".py"):
    """
    Generador que busca todos los archivos con una extensión dada
    en un directorio y sus subdirectorios.
    """
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                file_path = os.path.join(root, file)
                # Convertir la ruta del archivo a un nombre de módulo (ej. 'src/utils.py' -> 'src.utils')
                module_name = os.path.splitext(file_path)[0].replace(os.sep, '.')
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    yield module_name, content
                except Exception as e:
                    print(f"[WARN] No se pudo leer el archivo {file_path}: {e}")


def run_code_pretraining(root_directory, batch_size=20):
    """
    Analiza masivamente una base de código, extrae su estructura con AST
    y la inyecta en el Knowledge Graph de Prometheus.
    """
    print("="*60)
    print("INICIANDO MODO DE PRE-ENTRENAMIENTO SOBRE CÓDIGO")
    print(f"Directorio raíz a analizar: {root_directory}")
    print("="*60)

    try:
        prometheus = PrometheusAGI(force_genesis=True)
        code_stream = stream_code_files(root_directory)
        
        nodes_to_add = []
        edges_to_add = []
        processed_files = 0
        start_time = time.time()

        for module_name, code_content in code_stream:
            try:
                # 1. Parsear el código a un AST
                tree = ast.parse(code_content, filename=module_name)
                
                # 2. Visitar el árbol para extraer la estructura
                visitor = CodeVisitor(module_name)
                visitor.visit(tree)
                
                # 3. Acumular nodos y aristas
                nodes_to_add.extend(visitor.nodes)
                edges_to_add.extend(visitor.edges)
                
                processed_files += 1
            except SyntaxError as e:
                print(f"[WARN] Error de sintaxis en {module_name}, archivo omitido: {e}")
            except Exception as e:
                print(f"[ERROR] Error inesperado procesando {module_name}: {e}")

            # 4. Inyectar en Neo4j por lotes
            if processed_files % batch_size == 0:
                print(f"  ... Procesados {processed_files} archivos. Inyectando lote en Neo4j...")
                prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)
                nodes_to_add, edges_to_add = [], [] # Limpiar
                gc.collect()

        # Inyectar el último lote
        if nodes_to_add or edges_to_add:
            print("  ... Inyectando último lote en Neo4j.")
            prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)

        end_time = time.time()
        print("\n--- Pre-entrenamiento de código completado. ---")
        prometheus.shutdown()
        print(f"Se analizaron {processed_files} archivos en {end_time - start_time:.2f} segundos.")
        print("El Knowledge Graph ahora contiene la estructura de la base de código.")
        print("="*60)

    except Exception as e:
        print(f"[ERROR FATAL] Ocurrió un error durante el pre-entrenamiento de código:")
        import traceback
        traceback.print_exc()



def stream_corpus(filename):
    """
    Lee el archivo del corpus línea por línea y lo devuelve como un generador de artículos,
    evitando cargar todo el archivo en memoria.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        article_lines = []
        for line in f:
            if line.strip() == '---':
                if article_lines:
                    yield "".join(article_lines)
                article_lines = []
            else:
                article_lines.append(line)
        if article_lines: # Asegurarse de procesar el último artículo
            yield "".join(article_lines)


def run_massive_pretraining(batch_size=50):
    """
    Versión avanzada del pre-entrenamiento que procesa el corpus como un stream
    y extrae tripletas de conocimiento (Sujeto-Verbo-Objeto) para enriquecer el grafo.
    """
    print("="*60)
    print("INICIANDO MODO DE PRE-ENTRENAMIENTO MASIVO (S-V-O)")
    print("="*60)

    # 1. Preparar el corpus y la IA
    descargar_y_preparar_corpus()
    try:
        prometheus = PrometheusAGI(force_genesis=True)
        # Cargamos el modelo completo de spaCy para un mejor análisis de dependencias
        nlp = spacy.load("es_core_news_sm")
        print(f"Corpus '{CORPUS_FILENAME}' listo para ser procesado como stream.")
        
        start_time = time.time()
        article_stream = stream_corpus(CORPUS_FILENAME)
        
        nodes_to_add = []
        edges_to_add = []
        processed_articles = 0

        # 2. Procesar el stream de artículos
        for article_text in article_stream:
            lines = article_text.strip().split('\n', 1)
            if len(lines) < 2: continue

            main_topic = lines[0].replace("TEMA:", "").strip().lower()
            body = lines[1]
            
            # Añadir el nodo del artículo principal
            nodes_to_add.append({'id': main_topic, 'props': {'type': 'Artículo Principal'}})

            doc = nlp(body)

            # 3. Extracción de tripletas Sujeto-Verbo-Objeto
            for sent in doc.sents:
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                        objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "obj")]

                        if subjects and objects:
                            subject = subjects[0].text.lower()
                            obj = objects[0].text.lower()
                            
                            # --- MODIFICACIÓN CLAVE AQUÍ ---
                            # En lugar de solo el verbo, guardamos una frase más completa.
                            # Esto es una simplificación, una versión más avanzada analizaría la subfrase verbal.
                            relacion_frase = f"{token.lemma_} {objects[0].head.text}"
                            relacion_id = re.sub(r'[^a-zA-Z0-9_]', '', token.lemma_).upper()
                            # --- FIN DE LA MODIFICACIÓN ---

                            nodes_to_add.append({'id': subject, 'props': {'type': 'Concepto'}})
                            nodes_to_add.append({'id': obj, 'props': {'type': 'Concepto'}})
                            
                            edges_to_add.append({
                                'source': subject,
                                'target': obj,
                                'type': 'ALGUN_TIPO', 
                                'props': {'text': 'frase que describe la relacion', 'source_article': main_topic}
                            })


            processed_articles += 1
            
            # 4. Inyectar en Neo4j en lotes para máxima eficiencia
            if processed_articles % batch_size == 0:
                print(f"  ... Procesados {processed_articles} artículos. Inyectando lote de {len(nodes_to_add)} nodos y {len(edges_to_add)} aristas en Neo4j...")
                prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)
                nodes_to_add, edges_to_add = [], [] # Limpiar los lotes
                gc.collect() # Liberar memoria

        # Inyectar el último lote restante
        if nodes_to_add or edges_to_add:
            print(f"  ... Inyectando último lote de {len(nodes_to_add)} nodos y {len(edges_to_add)} aristas.")
            prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)

        end_time = time.time()
        print("\n--- Proceso de aprendizaje masivo completado. ---")
        prometheus.shutdown()
        print(f"Tiempo total del proceso: {end_time - start_time:.2f} segundos.")
        print(f"El Knowledge Graph ahora contiene relaciones estructuradas (Sujeto-Verbo-Objeto).")
        print("="*60)

    except Exception as e:
        print(f"[ERROR FATAL] Ocurrió un error durante el pre-entrenamiento masivo:")
        import traceback
        traceback.print_exc()

def run_optimized_pretraining(batch_size=100):
    """
    Versión optimizada que procesa artículos en lotes usando nlp.pipe
    y realiza inserciones masivas en Neo4j.
    """
    print("="*60)
    print("INICIANDO MODO DE PRE-ENTRENAMIENTO OPTIMIZADO DE PROMETHEUS")
    print("="*60)

    descargar_y_preparar_corpus()

    try:
        prometheus = PrometheusAGI(force_genesis=True)
        nlp = spacy.load("es_core_news_sm") # Usamos spaCy directamente

        with open(CORPUS_FILENAME, 'r', encoding='utf-8') as f:
            text_chunks = [paragraph for paragraph in f.read().split('\n\n---\n\n') if paragraph.strip()]
        print(f"Corpus '{CORPUS_FILENAME}' cargado. Se procesarán {len(text_chunks)} artículos en lotes de {batch_size}.")

        start_time = time.time()
        
        for i in range(0, len(text_chunks), batch_size):
            chunk_batch = text_chunks[i:i + batch_size]
            
            # Extraer títulos y cuerpos para procesar con nlp.pipe
            titles = [chunk.split('\n', 1)[0].replace("TEMA:", "").strip() for chunk in chunk_batch]
            bodies = [chunk.split('\n', 1)[1] if len(chunk.split('\n', 1)) > 1 else "" for chunk in chunk_batch]

            nodes_to_add = []
            edges_to_add = []

            # Usar nlp.pipe para un procesamiento mucho más rápido en paralelo (usa todos los cores de CPU)
            docs = nlp.pipe(bodies, n_process=-1)

            for doc, main_topic in zip(docs, titles):
                if not main_topic: continue

                # Añadir el nodo principal
                definition = " ".join([sent.text.strip() for sent in doc.sents][:2])
                nodes_to_add.append({'id': main_topic.lower(), 'props': {'definition': definition}})
                
                # Extraer y añadir conceptos relacionados
                related_concepts = set(chunk.text.lower() for chunk in doc.noun_chunks if 1 < len(chunk.text.split()) < 4)
                for concept in related_concepts:
                    if concept != main_topic.lower():
                        nodes_to_add.append({'id': concept, 'props': {}})
                        edges_to_add.append({
                            'source': main_topic.lower(), 
                            'target': concept, 
                            'type': 'related_to', 
                            'props': {'weight': 1.0}
                        })
            
            # Inyectar el lote completo en Neo4j
            if nodes_to_add or edges_to_add:
                prometheus.knowledge_graph.add_batch(nodes_to_add, edges_to_add)

            print(f"  ... Lote procesado. Artículos {i+batch_size}/{len(text_chunks)} inyectados en el Knowledge Graph.")

        end_time = time.time()
        print("\n--- Proceso de aprendizaje masivo completado. ---")
        prometheus.shutdown() # Guardar y cerrar conexiones
        print(f"\nEl Knowledge Graph ha sido actualizado con el conocimiento del corpus.")
        print(f"Tiempo total del proceso de inyección optimizado: {end_time - start_time:.2f} segundos.")
        print("="*60)

    except Exception as e:
        print(f"[ERROR] Ocurrió un error detallado durante el pre-entrenamiento optimizado:")
        import traceback
        traceback.print_exc()
# AÑADE esta nueva función a tu script

def bootstrap_kg(prometheus_instance):
        print("[KG BOOTSTRAP] Iniciando el bootstrap del Knowledge Graph...")
        kg = prometheus_instance.knowledge_graph
        nodes_to_create = [
            {"id": "perro", "name": "perro", "description": "Animal doméstico, mamífero, canino.", "labels": ["Concept", "Animal"]},
            {"id": "mamifero", "name": "mamífero", "description": "Clase de vertebrados de sangre caliente con glándulas mamarias.", "labels": ["Concept", "ClaseAnimal"]},
            {"id": "animal_domestico", "name": "animal doméstico", "description": "Animal que vive en relación con el ser humano, adaptado a la convivencia.", "labels": ["Concept", "TipoAnimal"]},
            {"id": "canino", "name": "canino", "description": "Perteneciente o relativo a la familia de los cánidos.", "labels": ["Concept", "ClaseAnimal"]}
        ]

        # Asegurarse de que el driver está inicializado
        if not kg.driver:
            kg.driver = kg._create_driver()
            if not kg.driver:
                print("[KG BOOTSTRAP ERROR] No se pudo inicializar el driver de Neo4j. Verifica la conexión.")
                return

        with kg.driver.session(database="neo4j") as session:
            try:
                # Crear nodos
                for node_data in nodes_to_create:
                    node_id = node_data['id'].lower().strip()
                    labels = ":".join(node_data['labels'])
                    properties = {k: v for k, v in node_data.items() if k not in ['id', 'labels']}
                    properties['id'] = node_id # Asegurar que 'id' también esté en properties

                    query = f"""
                        MERGE (n:{labels} {{id: $id}})
                        SET n += $properties
                        RETURN n
                    """
                    try:
                        session.run(query, id=node_id, properties=properties)
                        print(f"[KG BOOTSTRAP] Nodo '{node_id}' creado/actualizado.")
                    except Exception as e:
                        print(f"[KG BOOTSTRAP ERROR] Fallo al crear/actualizar nodo '{node_id}': {e}")
                        # Continuar con los demás nodos pero registrar el error

                # Crear relaciones
                relations_to_create = [
                    {"source": "perro", "target": "mamifero", "type": "ES_UN", "properties": {"text": "un perro es un tipo de mamífero", "relevance": 0.8}},
                    {"source": "perro", "target": "animal_domestico", "type": "ES_UN", "properties": {"text": "un perro es un animal doméstico", "relevance": 0.7}},
                    {"source": "mamifero", "target": "animal", "type": "ES_UNA_CLASE_DE", "properties": {"text": "mamífero es una clase de animal", "relevance": 0.9}},
                    {"source": "perro", "target": "canino", "type": "ES_UN", "properties": {"text": "el perro pertenece a la familia de los caninos", "relevance": 0.85}},
                ]

                for rel_data in relations_to_create:
                    source_id = rel_data['source'].lower().strip()
                    target_id = rel_data['target'].lower().strip()
                    rel_type = rel_data['type']
                    properties = rel_data['properties']
                    
                    query = f"""
                        MATCH (a:Concept {{id: $source_id}})
                        MATCH (b:Concept {{id: $target_id}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $properties
                        RETURN r
                    """
                    try:
                        session.run(query, source_id=source_id, target_id=target_id, properties=properties)
                        print(f"[KG BOOTSTRAP] Relación '{rel_type}' de '{source_id}' a '{target_id}' creada/actualizada.")
                    except Exception as e:
                        print(f"[KG BOOTSTRAP ERROR] Fallo al crear/actualizar relación {rel_type} entre '{source_id}' y '{target_id}': {e}")
                        # Continuar con las demás relaciones

                print("[KG BOOTSTRAP] Bootstrap del Knowledge Graph completado con los nodos base.")
            
            except Exception as e:
                print(f"[KG BOOTSTRAP FATAL ERROR] Fallo durante la sesión del bootstrap: {e}")
def run_lightweight_pretraining():
    """
    Carga el corpus local y lo usa para poblar el Knowledge Graph de Prometheus.
    """
    print("="*60)
    print("INICIANDO MODO DE PRE-ENTRENAMIENTO LIGERO DE PROMETHEUS")
    print("="*60)

    descargar_y_preparar_corpus()

    try:
        prometheus = PrometheusAGI(force_genesis=True)
        with open(CORPUS_FILENAME, 'r', encoding='utf-8') as f:
            text_chunks = [paragraph for paragraph in f.read().split('\n\n---\n\n') if paragraph.strip()]
        print(f"Corpus '{CORPUS_FILENAME}' cargado. Se procesarán {len(text_chunks)} artículos.")

        learn_gene = LearnFromTextGene(
            text_var="texto_a_aprender",
            graph=prometheus.knowledge_graph,
            nlp_processor=prometheus.intent_engine.nlp,
            similarity_model=prometheus.similarity_model
        )
        learning_chromosome = Chromosome(genes=[learn_gene], description="Estrategia de Inyección de Conocimiento")
        executor = ChromosomeExecutor()

        start_time = time.time()
        for i, chunk in enumerate(text_chunks):
            if (i + 1) % 100 == 0:
                 print(f"  Inyectando conocimiento del artículo {i+1}/{len(text_chunks)}...")

            lines = chunk.split('\n', 1)
            title_line = lines[0]
            body = lines[1] if len(lines) > 1 else ""

            title = title_line.replace("TEMA:", "").strip()

            context = ExecutionContext(initial_vars={
                "texto_a_aprender": body,
                "main_topic": title
            })
            
            executor.execute(learning_chromosome, context.memory)
        end_time = time.time()

        print("\n--- Proceso de aprendizaje completado. ---")
        prometheus._save_mind_state()
        print(f"\nEl Knowledge Graph ha sido actualizado con el conocimiento del corpus.")
        print(f"Tiempo total del proceso de inyección: {end_time - start_time:.2f} segundos.")
        print("="*60)

    except Exception as e:
        # --- BLOQUE MODIFICADO ---
        # Este nuevo bloque imprimirá un informe de error detallado.
        print(f"[ERROR] Ocurrió un error detallado durante el pre-entrenamiento:")
        import traceback
        traceback.print_exc()
        # --- FIN DEL BLOQUE MODIFICADO ---

def run_interactive_mode(prometheus: "PrometheusAGI"):
    print("\n" + "="*60 + "\nINICIANDO MODO INTERACTIVO...\n" + "="*60)
    while True:
        try:
            user_input = input("\nTU ORDEN > ").strip()
            if user_input.lower() in ['exit', 'quit', 'salir']: break
            ai_response = prometheus.think(user_input)
            print("\n" + "-"*45 + f"\nPROMETEO RESPONDE: {ai_response}\n" + "-"*45)
        except KeyboardInterrupt: break
    prometheus._save_mind_state()

def run_voice_interactive_mode(prometheus: "PrometheusAGI"):
    """Ejecuta una sesión interactiva continua basada en voz con Prometheus."""
    print("\n" + "="*60 + "\nINICIANDO MODO INTERACTIVO POR VOZ...\n" + "="*60)
    print("Di 'adiós' o 'salir' para terminar la sesión.")
    
    voice_gene = VoiceInteractionGene()

    voice_gene.speak("Hola, soy Prometheus. ¿En qué puedo ayudarte?")

    while True:
        try:
            user_input = voice_gene.listen()

            if not user_input:
                continue

            if user_input.lower() in ['adiós', 'salir', 'exit', 'quit']:
                voice_gene.speak("Hasta luego. Guardando mi estado mental.")
                break
            
            ai_response = prometheus.think(user_input)

            voice_gene.speak(ai_response or "He procesado tu solicitud, pero no he generado una respuesta textual.")

        except KeyboardInterrupt:
            voice_gene.speak("Interrupción detectada. Saliendo.")
            break
            
    prometheus._save_mind_state()
    print("\n" + "="*60 + "\nSESIÓN INTERACTIVA POR VOZ FINALIZADA.\n" + "="*60)
# EN: el final de tu archivo PROMETEUSV3.py
# REEMPLAZA tu bloque if __name__ == '__main__': existente con este código:

def main(args):
    """Función principal para encapsular la lógica de ejecución de Prometheus."""
    if torch.cuda.is_available():
        print(f"¡Éxito! PyTorch ha detectado una GPU compatible.")
        print(f"Nombre del dispositivo: {torch.cuda.get_device_name(0)}")
    else:
        print("Advertencia: PyTorch no pudo encontrar una GPU compatible con CUDA. Se ejecutará en CPU.")
        print("Para un rendimiento óptimo, asegúrate de haber instalado los drivers de NVIDIA y la versión de PyTorch para GPU.")

    prometheus_mind = None
    try:
        if args.mode == 'bootstrap':
            prometheus_mind = PrometheusAGI(force_genesis=True)
            bootstrap_kg(prometheus_mind)
        elif args.mode == 'pretrain':
            run_optimized_pretraining()
        elif args.mode == 'massive_pretrain':
            run_massive_pretraining()
        elif args.mode == 'code_pretrain':
            run_code_pretraining(root_directory=args.code_dir)
        elif args.mode == 'hf_code_pretrain':
            run_huggingface_code_pretraining(dataset_name=args.dataset)
        else:
            prometheus_mind = PrometheusAGI(force_genesis=args.force_genesis)

            if args.mode == 'marathon' or args.mode == 'exploratory':
                dojo = Dojo(prometheus_mind)
                if args.mode == 'marathon':
                    dojo.run_marathon(generations=args.generations, population_size=args.population)
                else:
                    dojo.run_exploratory_mode(num_episodes=args.episodes, generations_per_episode=args.generations, population_size=args.population)
            
            elif args.mode == 'simulation':
                env = GridWorldEnv()
                world_stream = WorldStream(environment=env)
                consciousness_loop = ConsciousnessLoop(prometheus_mind, world_stream)

                # --- LÍNEA CLAVE QUE EVITA EL ERROR ---
                prometheus_mind.env_size = 10 # O el tamaño que tenga tu GridWorldEnv
                # ------------------------------------

                print("[MAIN] Dando a Prometheus un objetivo inicial por curiosidad...")
                temp_ctx = ExecutionContext()
                CuriosityGene(prometheus_mind).execute(temp_ctx)

                try:
                    world_stream.start()
                    consciousness_loop.run()
                finally:
                    print("[MAIN] Deteniendo el stream del mundo...")
                    world_stream.stop()
                    world_stream.join(timeout=2)
                    if world_stream.is_alive():
                        world_stream.terminate()
                    print("[MAIN] Simulación finalizada.")


            elif args.mode == 'interactive':
                        # Asumiendo que has implementado la versión asíncrona de los pasos anteriores
                asyncio.run(run_interactive_mode_async(prometheus_mind))

            elif args.mode == 'voice':
                run_voice_interactive_mode(prometheus_mind)

    except Exception as e:
        print(f"\n[ERROR FATAL] Ocurrió un error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if prometheus_mind:
            prometheus_mind.shutdown()

if __name__ == '__main__':
    # Es recomendable usar 'spawn' para multiprocessing con PyTorch/CUDA en Windows o macOS.
    # multiprocessing.set_start_method('spawn', force=True) # Descomentar si tienes problemas
    multiprocessing.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser(description="PrometheusAGI - Una IA Evolutiva y Auto-expansiva")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        default='interactive', 
        choices=['marathon', 'exploratory', 'interactive', 'simulation', 'voice', 'bootstrap', 'pretrain', 'massive_pretrain', 'code_pretrain', 'hf_code_pretrain'],
        help='Modo de ejecución: marathon (entrenamiento), exploratory (auto-mejora), interactive (texto), simulation (entorno virtual), voice (interactivo por voz), pretrain (entrenamiento inicial del KG).'
    )
    
    parser.add_argument('--population', type=int, default=15, help='Tamaño de la población para la evolución.')
    parser.add_argument('--generations', type=int, default=8, help='Número de generaciones por evolución.')
    parser.add_argument('--episodes', type=int, default=50, help='Número de episodios para el modo exploratorio.')
    parser.add_argument('--force-genesis', action='store_true', help='Fuerza la regeneración de la mente desde cero (ignora archivos de estado guardados).')
    parser.add_argument(
        '--dataset',
        type=str,
        default='codeparrot/the-stack-sm', # Un buen valor por defecto
        help='Nombre del dataset de código en Hugging Face (ej. "codeparrot/the-stack-sm").'
    )
    parser.add_argument(
        '--code-dir',
        type=str,
        default='.',
        help='Directorio raíz para analizar código en el modo code_pretrain.'
    )

    # El "guard" ahora solo prepara los argumentos y llama a la función principal.
    # Esto es mucho más seguro para multiprocessing.
    args = parser.parse_args()
    main(args)