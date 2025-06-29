# prometheus_agi/knowledge/graph.py
import os
import re
from typing import List, Dict
from neo4j import GraphDatabase, basic_auth

# Importamos la configuración centralizada
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class KnowledgeGraph:
    """Implementación del KnowledgeGraph respaldada por Neo4j."""
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        print("[KG] Conectando al Knowledge Graph en Neo4j...")
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self._connect()

    def _connect(self):
        """Inicializa la conexión con la base de datos."""
        try:
            if hasattr(self, 'driver') and self.driver:
                self.driver.close()
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            self.driver.verify_connectivity()
            print(f"[KG pid={os.getpid()}] Conexión con Neo4j establecida.")
            self._create_constraints()
        except Exception as e:
            print(f"[ERROR FATAL pid={os.getpid()}] No se pudo conectar a Neo4j: {e}")
            raise

    def __getstate__(self):
        """Excluye el driver no serializable para multiprocessing."""
        state = self.__dict__.copy()
        del state['driver']
        return state

    def __setstate__(self, state):
        """Recrea la conexión en un nuevo proceso."""
        self.__dict__.update(state)
        self._connect()

    def _create_constraints(self):
        """Asegura que los nodos de conceptos sean únicos."""
        with self.driver.session(database="neo4j") as session:
            session.run("CREATE CONSTRAINT concept_uniqueness IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")

    def create_vector_index(self):
        """Crea un índice vectorial en Neo4j para búsquedas de similitud."""
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
            print(f"[AVISO] No se pudo crear el índice vectorial (requiere Neo4j 5.11+ con GDS): {e}")

    def find_similar_nodes(self, vector: List[float], top_k: int = 5) -> List[dict]:
        """Encuentra los 'top_k' nodos más similares a un vector dado."""
        query = """
        CALL db.index.vector.queryNodes('concept_embeddings', $top_k, $vector)
        YIELD node, score
        RETURN node.id AS id, node.definition AS definition, score
        """
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, top_k=top_k, vector=vector)
                return [record.data() for record in result]
        except Exception:
            # Fallback si el índice no existe
            return []
            
    def add_batch(self, nodes: List[Dict], edges: List[Dict]):
        """Añade nodos y aristas a Neo4j en un lote masivo."""
        with self.driver.session(database="neo4j") as session:
            if nodes:
                session.run("""
                    UNWIND $nodes AS node_data
                    MERGE (c:Concept {id: node_data.id})
                    ON CREATE SET c += node_data.props
                    ON MATCH SET c += node_data.props
                """, nodes=nodes)

            if edges:
                for edge in edges:
                    rel_type_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', edge['type'])
                    if not rel_type_cleaned: continue
                    if rel_type_cleaned[0].isdigit():
                        rel_type_cleaned = "rel_" + rel_type_cleaned
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

    def get_outgoing_relations(self, node_id: str) -> List[Dict]:
        """Recupera todas las relaciones salientes de un nodo."""
        node_id_lower = node_id.lower().strip()
        query = """
            MATCH (n:Concept {id: $node_id})-[r]->(m:Concept)
            RETURN type(r) as rel_type, properties(r) as properties, m.id as target_node_id
            """
        try:
            with self.driver.session(database="neo4j") as session:
                result = session.run(query, node_id=node_id_lower)
                return [record.data() for record in result]
        except Exception as e:
            print(f"[ERROR KG] No se pudieron obtener las relaciones para '{node_id_lower}': {e}")
            return []
            
    # Añadir el resto de métodos: add_node_if_not_exists, get_node_definition, etc.
    # ... (El código es idéntico al del script original)
    def add_node_with_definition(self, node_id: str, definition: str):
        if definition and len(definition) > 10:
            self.add_node_if_not_exists(node_id, definition=definition)
        else:
            self.add_node_if_not_exists(node_id)

    def add_node_if_not_exists(self, node_id: str, **attrs):
        node_id_lower = node_id.lower().strip()
        with self.driver.session(database="neo4j") as session:
            session.run("MERGE (c:Concept {id: $id}) SET c += $props", id=node_id_lower, props=attrs)

    def add_edge(self, source_id: str, target_id: str, relationship_type: str, **attrs):
        source_id_lower = source_id.lower().strip()
        target_id_lower = target_id.lower().strip()
        rel_type_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', relationship_type).upper()
        if not rel_type_cleaned: return
        with self.driver.session(database="neo4j") as session:
            query = f"""
                MATCH (source:Concept {{id: $source_id}})
                MATCH (target:Concept {{id: $target_id}})
                MERGE (source)-[r:{rel_type_cleaned}]->(target)
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
        start_id, end_id = start_node_id.lower().strip(), end_node_id.lower().strip()
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

    def close(self):
        """Cierra la conexión con la base de datos."""
        if self.driver:
            self.driver.close()
            print("[KG] Conexión con Neo4j cerrada.")