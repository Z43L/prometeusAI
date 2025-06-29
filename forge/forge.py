# prometheus_agi/forge/forge.py
# La lógica para la forja de genes (GeneForge) es más avanzada y se introduce
# en versiones posteriores. En V5, esta clase es un placeholder.
# La funcionalidad se simula o se deja para desarrollo futuro.

from typing import TYPE_CHECKING
from forge.oracle import LLMCodeGenerator
from config import GOOGLE_API_KEY
import os
import json
from config import SANDBOX_DIR, CUSTOM_GENES_DIR

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI

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
        self.llm = self.external_oracle.model # Acceso directo al modelo LLM de Gemini
        # Oráculo Interno (Aprendiz), para intentar generar soluciones por sí mismo.
        self.internal_oracle = None
        internal_oracle_path = "custom_genes/internal_oracle"
        if os.path.exists(internal_oracle_path):
            print(f"[FORGE] Cargando Oráculo Interno desde '{internal_oracle_path}'...")
            # En una implementación real, aquí se cargaría el modelo afinado con PEFT.
            # self.internal_oracle = AutoModelForCausalLM.from_pretrained(...)
            print("[FORGE] Oráculo Interno cargado (simulado).")
        
        os.makedirs(SANDBOX_DIR, exist_ok=True)
        os.makedirs(CUSTOM_GENES_DIR, exist_ok=True)

    def attempt_new_gene_creation(self, intent: str, sample_query: str):
        """
        El método principal que orquesta el ciclo completo de creación de un nuevo gen.
        """
        print(f"\n===== CICLO DE GÉNESIS DE GENES INICIADO PARA LA INTENCIÓN '{intent}' =====")
        
        # Paso 1: Investigación (simplificada)
        research_summary = f"La investigación sobre la intención '{intent}' sugiere la necesidad de una solución programática específica. Consulta de ejemplo: '{sample_query}'"
        gene_name = f"{intent.capitalize()}Gene"
        prompt = self._create_gene_generation_prompt(intent, research_summary, gene_name)

        # --- Jerarquía de Oráculos ---
        # Paso 2: Intentar generar con el oráculo interno primero
        gene_code = self._generate_with_internal_oracle(prompt)
        source = "Interno"
        
        if not gene_code:
            print("[FORGE] El Oráculo Interno no pudo generar una solución. Consultando al Oráculo Externo (Gemini)...")
            gene_code = self.external_oracle.generate_code(prompt)
            source = "Externo (Gemini)"

        if not gene_code:
            print("[FORGE] Ningún oráculo pudo generar código. Abortando génesis.")
            return

        print(f"[FORGE] Código del gen generado por el Oráculo {source}.")
        
        # Paso 3: Ensamblar el fichero de código
        temp_gene_path = self._assemble_gene_file(gene_name, gene_code)
        
        # Paso 4: Generar y ejecutar pruebas en un sandbox seguro
        test_code = self.external_oracle.generate_test_code(gene_code, intent, sample_query)
        test_passed = self._test_in_sandbox(temp_gene_path, test_code)

        # Paso 5: Integrar el gen si las pruebas son exitosas
        if test_passed:
            print(f"[FORGE] ¡Prueba superada! Integrando '{gene_name}' en el genoma de Prometheus...")
            # Si el éxito vino del oráculo externo, lo guardamos como lección para el interno.
            if source == "Externo (Gemini)":
                self._save_successful_genesis(prompt, gene_code)
            self._integrate_gene(temp_gene_path, gene_name)
        else:
            print(f"[FORGE] La prueba del gen falló. Descartando el gen.")
            
        print("====================== GÉNESIS DE GENES FINALIZADO ======================\n")

    def _generate_with_internal_oracle(self, prompt: str) -> str | None:
        """Intenta generar código usando el modelo local afinado."""
        if not self.internal_oracle:
            return None
        print("[FORGE] Intentando generar gen con el Oráculo Interno...")
        # Lógica real para generar código con el modelo local iría aquí.
        return None # Simulación de fallo para forzar el uso del oráculo externo en esta demo.

    def _save_successful_genesis(self, prompt: str, code: str):
        """Añade un ejemplo exitoso de prompt->código al corpus de entrenamiento."""
        try:
            with open(self.genesis_corpus_path, "a", encoding="utf-8") as f:
                record = {"prompt": prompt, "completion": code}
                f.write(json.dumps(record) + "\n")
            print(f"[FORGE] Lección de génesis guardada en '{self.genesis_corpus_path}'.")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar la lección de génesis: {e}")

    def _create_gene_generation_prompt(self, intent, research, gene_name) -> str:
        """Crea un prompt detallado y bien estructurado para guiar al LLM."""
        # Este prompt es crucial para la calidad del código generado.
        # Incluye el contexto de la arquitectura para que el LLM sepa cómo debe ser un "Gen".
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
    def __init__(self, initial_vars=None): self.memory = initial_vars or {{}}
    def set(self, key, value): self.memory[key] = value
    def get(self, key, default=None): return self.memory.get(key, default)
"""
    