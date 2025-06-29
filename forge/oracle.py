# prometheus_agi/forge/oracle.py
import re
import google.generativeai as genai

# Importaciones del proyecto
from config import GOOGLE_API_KEY, GEMINI_MODEL_NAME

class LLMCodeGenerator:
    """Se comunica con Google Gemini para generar código para nuevos genes."""
    def __init__(self, api_key: str = GOOGLE_API_KEY):
        if not api_key or api_key == "AIzaSyC8gKreBW0CNtEYcvfU5FH3g_Q-LJURpq8":
            print("[WARN] LLMCodeGenerator: La clave de API de Google no está configurada. La génesis de genes no funcionará.")
            self.model = None
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print(f"[LLM_CODE_GEN] Conectado a Gemini {GEMINI_MODEL_NAME}.")
        except Exception as e:
            print(f"[ERROR] No se pudo configurar la API de Gemini: {e}")
            self.model = None

    def _extract_python_code(self, text: str) -> str:
        """Extrae el bloque de código Python de la respuesta del LLM."""
        match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def generate_code(self, prompt: str) -> str:
        """Envía un prompt a Gemini y extrae el código de la respuesta."""
        if not self.model: return ""
        try:
            response = self.model.generate_content(prompt)
            return self._extract_python_code(response.text)
        except Exception as e:
            print(f"[ERROR] Falló la llamada a la API de Gemini: {e}")
            return ""