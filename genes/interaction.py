# prometheus_agi/genes/interaction.py
import pyttsx3
import speech_recognition as sr
import spacy
import random
# Importaciones del proyecto
from core.base import Gene, ExecutionContext

LEXICON_BRAIN_V2 = {
    "EXPRESS_EMPATHY": [{"text": "Comprendo que la situación pueda ser compleja.", "style": {"formality": "formal"}}, {"text": "Eso suena fatal, de verdad.", "style": {"formality": "informal"}}],
    "OFFER_HELP": [{"text": "¿Hay algo en lo que pueda asistirle?", "style": {"formality": "formal"}}, {"text": "¿Quieres que te dé un ejemplo?", "style": {"formality": "informal"}}],
    "GREETING": [{"text": "Hola. ¿En qué puedo ayudarte?", "style": {"formality": "formal"}}, {"text": "¡Hola! Dime qué necesitas.", "style": {"formality": "informal"}}]
}

class VoiceInteractionGene(Gene):
    """Permite a Prometheus escuchar y hablar."""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty("rate", 160)

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
            self.speak("Hubo un error con el servicio de voz.")
            return ""

    def speak(self, text: str):
        print(f"[🗣️ Prometheus]: {text}")
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def execute(self, context: ExecutionContext):
        response = context.get_final_response()
        if response:
            self.speak(response)
        else:
            user_input = self.listen()
            context.set("query", user_input)

class DetectSentimentGene(Gene):
    """Detecta el sentimiento en la consulta del usuario."""
    def execute(self, context: ExecutionContext):
        query = context.get("query", "").lower()
        if any(w in query for w in ["fatal", "odio", "triste"]):
            sentiment = "negative"
        elif any(w in query for w in ["gracias", "genial", "perfecto"]):
            sentiment = "positive"
        else:
            sentiment = "neutral"
        context.set("user_sentiment", sentiment)

class StylizedResponseGene(Gene):
    """Genera una respuesta basada en un léxico y el estilo del usuario."""
    def __init__(self, intent: str):
        self.intent = intent

    def execute(self, context: ExecutionContext):
        style = context.get("user_style_profile", {"formality": "formal"})
        options = LEXICON_BRAIN_V2.get(self.intent, [])
        if options:
            matches = [o for o in options if o["style"]["formality"] == style["formality"]]
            context.set_final_response(random.choice(matches if matches else options)["text"])

class StyleAnalyzerGene(Gene):
    """Analiza el estilo de la consulta del usuario."""
    def execute(self, context: ExecutionContext):
        text = context.get("query", "").lower()
        formality = "informal" if any(w in text for w in ["tío", "osea", "que tal"]) else "formal"
        context.set("user_style_profile", {"formality": formality})

class AnalizadorGramaticalGene(Gene):
    """Analiza gramaticalmente una frase usando spaCy."""
    _fallback_nlp = None

    def __init__(self, nlp_processor):
        self.nlp = nlp_processor

    def execute(self, context: ExecutionContext):
        frase = context.get("query")
        if not frase:
            context.set_final_response("No me has proporcionado una frase para analizar.")
            return
        
        try:
            # Intenta usar el nlp del worker si está disponible
            doc = self.nlp(frase)
        except NameError:
            if AnalizadorGramaticalGene._fallback_nlp is None:
                AnalizadorGramaticalGene._fallback_nlp = spacy.load("es_core_news_sm")
            doc = AnalizadorGramaticalGene._fallback_nlp(frase)

        reporte = ["Análisis gramatical de la frase:"]
        reporte.append("-" * 40)
        for token in doc:
            reporte.append(f"'{token.text}' -> '{token.lemma_}' | {token.pos_} | {token.dep_}")
        
        context.set_final_response("\n".join(reporte))