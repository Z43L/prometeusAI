# prometheus_agi/genes/system.py
import spacy
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from sentence_transformers import util

# Importaciones del proyecto
from core.base import Gene, ExecutionContext
from knowledge.graph import KnowledgeGraph

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI

class CognitiveCoreGene(Gene):
    """Orquesta la consulta al KG y el uso de herramientas."""
    def __init__(self, mind: 'PrometheusAGI'):
        # Usamos un type hint como string para evitar la importación circular
        self.mind = mind

    def _select_best_tool(self, query: str) -> Gene | None:
        """Selecciona la mejor herramienta para la consulta."""
        query_embedding = self.mind.similarity_model.encode(query)
        best_tool_name, highest_score = None, -1.0
        for descriptor in self.mind.tool_descriptors:
            desc_embedding = self.mind.similarity_model.encode(descriptor.purpose_description)
            score = util.pytorch_cos_sim(query_embedding, desc_embedding).item()
            if score > highest_score:
                highest_score, best_tool_name = score, descriptor.gene_class.__name__
        
        if highest_score > 0.4:
            return self.mind.tool_genes.get(best_tool_name)
        return None

    def _analyze_query(self, query: str) -> Tuple[List[float], List[str]]:
        """Convierte la consulta en un vector y extrae keywords."""
        query_vector = self.mind.similarity_model.encode(query).tolist()
        doc = self.mind.lightweight_nlp(query)
        keywords = [ent.text.lower() for ent in doc.ents]
        for chunk in doc.noun_chunks:
            # Filtra palabras vacías y muy cortas
            if chunk.text.lower() not in self.mind.lightweight_nlp.Defaults.stop_words and len(chunk.text) > 2:
                keywords.append(chunk.text.lower())
        return query_vector, list(set(keywords))

    def _find_start_nodes(self, vector: List[float], keywords: List[str]) -> List[dict]:
        """Encuentra nodos de partida en el KG por similitud y keywords."""
        similar_nodes = self.mind.knowledge_graph.find_similar_nodes(vector, top_k=3)
        found_nodes = {node['id']: node for node in similar_nodes}
        for keyword in keywords:
            if keyword not in found_nodes and self.mind.knowledge_graph.has_node(keyword):
                found_nodes[keyword] = {'id': keyword, 'score': 0.99} # Alta prioridad si es una keyword directa
        return sorted(found_nodes.values(), key=lambda x: x.get('score', 0), reverse=True)

    def execute(self, context: ExecutionContext):
        query = context.get("query")
        if not query: return
        print("[CognitiveCore] Iniciando ciclo de pensamiento...")
        
        # 1. Analizar la consulta y buscar en la memoria interna (KG)
        query_vector, keywords = self._analyze_query(query)
        start_nodes = self._find_start_nodes(query_vector, keywords)
        synthesizer = GraphTraversalSynthesizer(self.mind)
        internal_answer = synthesizer.synthesize_answer(query, start_nodes)
        
        final_answer = internal_answer
        
        # 2. Decidir si se necesita una herramienta externa
        if not internal_answer or len(internal_answer) < 60:
            tool_gene = self._select_best_tool(query)
            if tool_gene:
                print(f"[CognitiveCore] Usando herramienta: {tool_gene.__class__.__name__}")
                tool_gene.execute(context)
                tool_result = context.get('web_summary') or context.get('scientific_summary') or context.get('calculation_result')
                if tool_result:
                    final_answer = f"Según mis herramientas: {tool_result}"
                else:
                    final_answer = "Usé mis herramientas, pero no encontré una respuesta clara. " + (internal_answer or "")
        
        context.set_final_response(final_answer)

class GraphTraversalSynthesizer:
    """Genera texto coherente "paseando" por el KG."""
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind

    def _get_most_relevant_path(self, start_node_id: str, query_embedding: np.ndarray) -> List[dict] | None:
        relations = self.mind.knowledge_graph.get_outgoing_relations(start_node_id)
        if not relations: return None

        best_path, highest_score = None, -1.0
        for rel in relations:
            # Construye un texto descriptivo para la relación
            relation_text = rel.get('properties', {}).get('text', rel.get('rel_type', '')).replace('_', ' ')
            target_node = rel.get('target_node_id', '')
            full_relation_text = f"{start_node_id} {relation_text} {target_node}"

            # Compara la consulta con el significado de la relación completa
            relation_embedding = self.mind.similarity_model.encode(full_relation_text)
            score = util.pytorch_cos_sim(query_embedding, relation_embedding).item()
            if score > highest_score:
                highest_score = score
                best_path = [
                    {'type': 'node', 'value': start_node_id},
                    {'type': 'relation', 'value': relation_text},
                    {'type': 'node', 'value': target_node}
                ]
        # Devuelve el camino solo si es suficientemente relevante
        return best_path if highest_score > 0.35 else None

    def synthesize_answer(self, query: str, start_nodes: List[dict]) -> str:
        if not start_nodes: return ""
        
        query_embedding = self.mind.similarity_model.encode(query)
        sentences = []
        # Intenta construir una frase para los nodos más relevantes
        for start_node in start_nodes[:2]:
            path = self._get_most_relevant_path(start_node['id'], query_embedding)
            if path:
                sentence = " ".join(p['value'] for p in path if p['value']).capitalize() + "."
                sentences.append(sentence)
        
        return " ".join(sentences)

class GetNodeDefinitionGene(Gene):
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph
    
    def execute(self, context: ExecutionContext):
        concept = context.get("main_topic")
        if not concept: return
        definition = self.graph.get_node_definition(concept)
        context.set("definition_text", definition)

class LearnFromTextGene(Gene):
    def __init__(self, graph: KnowledgeGraph, nlp_processor):
        self.graph = graph
        self.nlp = nlp_processor
    
    def execute(self, context: ExecutionContext):
        text = context.get("text_to_learn")
        topic = context.get("main_topic")
        if not text or not topic: return

        print(f"  [LearnGene] Aprendiendo sobre '{topic}'...")
        doc = self.nlp(text)
        # Usa las dos primeras frases como definición
        definition = " ".join([sent.text.strip() for sent in doc.sents][:2])
        self.graph.add_node_with_definition(topic, definition)
        
        # Extrae conceptos relacionados
        related_concepts = {chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) < 4 and chunk.text.lower() != topic}
        for concept in related_concepts:
            self.graph.add_node_if_not_exists(concept)
            self.graph.add_edge(topic, concept, "related_to")

class DeepReasoningGene(Gene):
    def __init__(self, graph: KnowledgeGraph, nlp_processor):
        self.graph = graph
        self.nlp = nlp_processor
    
    def execute(self, context: ExecutionContext):
        query = context.get("query", "")
        doc = self.nlp(query)
        # Extrae conceptos ignorando palabras de unión
        concepts = [c.text.lower() for c in doc.noun_chunks if c.text.lower() not in ["relación", "entre", "y", "cuál es la"]]
        if len(concepts) < 2:
            context.set("reasoning_result", "Necesito al menos dos conceptos claros para encontrar una relación.")
            return

        c1, c2 = concepts[0], concepts[1]
        if not self.graph.has_node(c1) or not self.graph.has_node(c2):
            context.set("reasoning_result", "No tengo suficiente información sobre uno o ambos conceptos.")
            return

        # 1. Buscar camino directo
        path = self.graph.find_shortest_path(c1, c2)
        if path and 1 < len(path) < 5:
            context.set("reasoning_result", f"He encontrado una conexión directa: {' -> '.join(path)}.")
            return

        # 2. Buscar vecinos en común
        n1 = set(self.graph.get_all_neighbors(c1))
        n2 = set(self.graph.get_all_neighbors(c2))
        common = n1.intersection(n2)
        if common:
            context.set("reasoning_result", f"No están conectados directamente, pero ambos se relacionan con: {', '.join(list(common)[:3])}.")
            return

        context.set("reasoning_result", "No encontré una relación clara entre ellos en mi base de conocimiento.")

class InferDefinitionFromNeighborsGene(Gene):
    def __init__(self, graph: KnowledgeGraph):
        self.graph = graph

    def execute(self, context: ExecutionContext):
        if context.get("definition_text"): return
        topic = context.get("main_topic")
        if not topic: return

        neighbors = self.graph.get_all_neighbors(topic)
        if not neighbors: return

        parts = []
        for neighbor in neighbors[:3]:
            neighbor_def = self.graph.get_node_definition(neighbor)
            if neighbor_def:
                parts.append(f"se relaciona con '{neighbor}', que se define como: {neighbor_def}")
        
        if parts:
            context.set("definition_text", f"No tengo una definición directa para '{topic}', pero puedo inferir que: {'. '.join(parts)}")

class FormulateResponseGene(Gene):
    def execute(self, context: ExecutionContext):
        if context.get_final_response(): return
        
        priority_keys = ["definition_text", "scientific_summary", "web_summary", "reasoning_result", "calculation_result", "failure_report", "plan_explanation"]
        
        for key in priority_keys:
            if result := context.get(key):
                context.set_final_response(str(result))
                return
        
        context.set_final_response("He procesado tu solicitud, pero no he generado una respuesta concluyente.")

class LinguisticAnalysisGene(Gene):
    def __init__(self, nlp_processor):
        self.nlp = nlp_processor

    def execute(self, context: ExecutionContext):
        frase = context.get("query")
        if not frase: 
            context.set_final_response("No hay frase para analizar.")
            return
        
        doc = self.nlp(frase)
        reporte = [f"Análisis gramatical para la frase '{frase}':", "-"*40]
        reporte.append(f"{'PALABRA':<15} | {'LEMA':<15} | {'CATEGORÍA':<10} | {'FUNCIÓN':<10}")
        reporte.append("-" * 60)
        for token in doc:
            reporte.append(f"{token.text:<15} | {token.lemma_:<15} | {token.pos_:<10} | {token.dep_:<10}")
        
        context.set_final_response("\n".join(reporte))