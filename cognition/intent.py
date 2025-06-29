# prometheus_agi/cognition/intent.py
import spacy
from transformers import pipeline
from typing import Dict, Any

# Importaciones del proyecto
from config import ZERO_SHOT_MODEL_NAME, SPACY_MODEL_NAME

class DynamicIntentEngine:
    """Analiza una consulta para extraer su intención y entidades clave."""
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
            print(f"\n[INTENT_ENGINE] Cargando modelo de clasificación zero-shot '{ZERO_SHOT_MODEL_NAME}'...")
            self._classification_pipeline = pipeline("zero-shot-classification", model=ZERO_SHOT_MODEL_NAME)
        return self._classification_pipeline

    @property
    def nlp(self):
        """Carga el modelo de spaCy solo cuando se accede por primera vez."""
        if self._nlp is None:
            print(f"\n[INTENT_ENGINE] Cargando modelo de spaCy '{SPACY_MODEL_NAME}'...")
            self._nlp = spacy.load(SPACY_MODEL_NAME)
        return self._nlp

    def _get_main_topic(self, text: str, intent: str) -> str:
        """Extrae el tópico principal de una consulta."""
        text_lower = text.lower().strip()
        prefixes_to_strip = {
            'DEFINIR': ["define el concepto de", "define la idea de", "define", "qué es", "que es", "cuál es el significado de"],
            'BUSCAR_WEB': ["busca en internet sobre", "busca sobre", "investiga sobre"],
            'BUSCAR_CIENCIA': ["busca un paper sobre", "encuentra artículos de"],
            'ANALIZAR_CODIGO': ["analiza el código de", "analiza el gen"],
        }
        
        if intent in prefixes_to_strip:
            for prefix in prefixes_to_strip[intent]:
                if text_lower.startswith(prefix):
                    return text_lower[len(prefix):].strip("?¿ '\"")
        
        doc = self.nlp(text)
        noun_chunks = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ == "NOUN"]
        return max(noun_chunks, key=len) if noun_chunks else text

    def analyze(self, query: str) -> Dict[str, Any]:
        """Procesa la consulta para determinar la intención y extraer entidades."""
        query_lower = query.lower()
        forced_intent = None
        intent_keywords = {
            "SALUDAR": ["hola", "buenas", "buenos días", "qué tal"],
            "RECORDAR": ["repite", "qué dijiste", "lo anterior"],
            "BUSCAR_CIENCIA": ["paper", "artículo científico", "pubmed"],
            "ANALIZAR_CODIGO": ["analiza el código", "explícame el gen"],
            "CALCULAR": ["cuánto es", "calcula", "suma", "resta"],
        }
        for intent_key, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                forced_intent = intent_key
                break
        
        if forced_intent:
            intent, confidence = forced_intent, 1.0
        else:
            analysis_result = self.classification_pipeline(query, self.candidate_labels, hypothesis_template="Esta frase es una {}.")
            intent = analysis_result['labels'][0]
            confidence = analysis_result['scores'][0]

        main_topic = self._get_main_topic(query, intent)
        doc = self.nlp(query)
        entities = [ent.text for ent in doc.ents]

        print(f"  [Intent Analysis] Query: '{query[:30]}...' -> Intent: {intent} ({confidence:.2f}), Topic: '{main_topic}'")
        return {"intent": intent, "confidence": confidence, "main_topic": main_topic, "entities": entities, "full_query": query}