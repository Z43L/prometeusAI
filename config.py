# prometheus_agi/config.py
import os

# --- Claves de API y Servicios ---
# Es MUY recomendable cargar esto desde variables de entorno en producción.
# export GOOGLE_API_KEY="tu_clave"
# export GOOGLE_CX="tu_cx"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "AIzaSyC8gKreBW0CNtEYcvfU5FH3g_Q-LJURpq8")
GOOGLE_CX = os.environ.get("GOOGLE_CX", "22b0b9b4060f94ccc")

# --- Conexión a Base de Datos ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "prometheus_password"

# --- Directorios ---
FORGE_DIR = "gene_forge"
MIND_STATE_DIR = "mind_state"
SANDBOX_DIR = "sandbox"
CUSTOM_GENES_DIR = "custom_genes"

# --- Rutas de Archivos ---
PATHS = {
    "kg": os.path.join(MIND_STATE_DIR, "knowledge_graph.pkl"),
    "patterns": os.path.join(MIND_STATE_DIR, "entity_patterns.pkl"),
    "profile": os.path.join(MIND_STATE_DIR, "fitness_profile.pkl"),
    "archive": os.path.join(MIND_STATE_DIR, "strategies_archive.pkl"),
    "dojo_dataset": "dojo_dataset.json",
    "specialists_archive": os.path.join(MIND_STATE_DIR, "specialists_archive.pkl"),
    "general_archive": os.path.join(MIND_STATE_DIR, "general_archive.pkl"),
    "fitness_profiles": os.path.join(MIND_STATE_DIR, "fitness_profiles.pkl"),
    "self_model": os.path.join(MIND_STATE_DIR, "self_model.pkl"),
    "gene_usage": os.path.join(MIND_STATE_DIR, "gene_usage.pkl"),
    "episodic_memory": os.path.join(MIND_STATE_DIR, "episodic_memory.log"),
    "genesis_corpus": os.path.join(FORGE_DIR, "genesis_corpus.jsonl"),
    "corpus_de_conocimiento": "corpus_de_conocimiento.txt",
}

# --- Parámetros del Dojo y Evolución ---
BATCH_SIZE = 1
EVOLUTION_FOCUS_SIZE = 1
MAX_SPECIALISTS_PER_INTENT = 5
PROMETHEUS_CANDIDATE_PERCENTILE = 0.4
STAGNATION_LIMIT = 4
INCAPACITY_THRESHOLD = 300
INCAPACITY_ATTEMPTS = 3

# --- Parámetros de Pre-entrenamiento ---
CORPUS_FILENAME = "corpus_wikipedia_es.txt"
NUM_ARTICULOS_A_PROCESAR = 1000

# --- Modelos de IA ---
SIMILARITY_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
SPACY_MODEL_NAME = 'es_core_news_sm'
ZERO_SHOT_MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
GEMINI_MODEL_NAME = 'gemini-1.5-pro-latest'

# --- Pre-entrenamiento ---
NUM_ARTICULOS_A_PROCESAR = 5000