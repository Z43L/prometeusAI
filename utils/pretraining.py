# prometheus_agi/utils/pretraining.py
import os
import gc
import re
import json
import time
import spacy
from datasets import load_dataset

# Importaciones del proyecto
from config import PATHS, CORPUS_FILENAME, NUM_ARTICULOS_A_PROCESAR
from knowledge.analysis import CodeVisitor

def setup_environment():
    """Crea un entorno de prueba si no existe."""
    if not os.path.exists(PATHS["dojo_dataset"]):
        print(f"Creando dataset de ejemplo '{PATHS['dojo_dataset']}'...")
        dummy_dataset = [
            {"INSTRUCTION": "¿Qué es la gravedad?", "RESPONSE": "La gravedad es la fuerza de atracción entre objetos con masa."},
            {"INSTRUCTION": "Define el concepto de 'democracia'.", "RESPONSE": "La democracia es un sistema de gobierno donde el poder reside en el pueblo."},
            {"INSTRUCTION": "Busca en internet sobre 'el futuro de la inteligencia artificial'", "RESPONSE": "La IA avanza hacia modelos más generales y éticos."}
        ]
        with open(PATHS["dojo_dataset"], 'w', encoding='utf-8') as f:
            json.dump(dummy_dataset, f, indent=2, ensure_ascii=False)

def descargar_y_preparar_corpus():
    """Descarga y prepara el corpus de Wikipedia si no existe."""
    if os.path.exists(CORPUS_FILENAME):
        return
    print("Corpus no existe localmente. Descargando de Hugging Face...")
    try:
        dataset = load_dataset("wikimedia/wikipedia", "20231101.es", streaming=True, split="train")
        subset = dataset.take(NUM_ARTICULOS_A_PROCESAR)
        with open(CORPUS_FILENAME, 'w', encoding='utf-8') as f:
            for i, article in enumerate(subset):
                text = re.sub(r'==.*?==', '', article['text'])
                text = re.sub(r'\n+', '\n\n', text).strip()
                f.write(f"TEMA: {article['title']}\n{text}\n\n---\n\n")
                if (i + 1) % 100 == 0:
                    print(f"  ... {i + 1}/{NUM_ARTICULOS_A_PROCESAR} artículos guardados.")
    except Exception as e:
        print(f"[ERROR FATAL] No se pudo descargar el corpus: {e}")
        exit()

def run_optimized_pretraining(prometheus, batch_size=100):
    """Pre-entrenamiento optimizado usando nlp.pipe."""
    print("="*60 + "\nINICIANDO MODO DE PRE-ENTRENAMIENTO OPTIMIZADO\n" + "="*60)
    descargar_y_preparar_corpus()
    nlp = spacy.load("es_core_news_sm")
    with open(CORPUS_FILENAME, 'r', encoding='utf-8') as f:
        chunks = [p for p in f.read().split('\n\n---\n\n') if p.strip()]
    
    start_time = time.time()
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        titles = [c.split('\n', 1)[0].replace("TEMA:", "").strip() for c in batch]
        bodies = [c.split('\n', 1)[1] if len(c.split('\n', 1)) > 1 else "" for c in batch]
        
        nodes, edges = [], []
        docs = nlp.pipe(bodies, n_process=-1)

        for doc, topic in zip(docs, titles):
            if not topic: continue
            definition = " ".join([sent.text.strip() for sent in doc.sents][:2])
            nodes.append({'id': topic.lower(), 'props': {'definition': definition}})
            concepts = set(c.text.lower() for c in doc.noun_chunks if 1 < len(c.text.split()) < 4)
            for concept in concepts:
                if concept != topic.lower():
                    nodes.append({'id': concept, 'props': {}})
                    edges.append({'source': topic.lower(), 'target': concept, 'type': 'related_to'})
        
        if nodes or edges:
            prometheus.knowledge_graph.add_batch(nodes, edges)
        print(f"  ... Lote procesado. Artículos {i+batch_size}/{len(chunks)} inyectados.")

    print(f"\nProceso completado en {time.time() - start_time:.2f} segundos.")