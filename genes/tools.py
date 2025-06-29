# prometheus_agi/genes/tools.py
import re
import requests
import ast
import operator as op
from bs4 import BeautifulSoup
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

from core.base import Gene, ExecutionContext

# Diccionario de operaciones seguras para la calculadora
SAFE_OPERATORS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.BitXor: op.xor,
    ast.USub: op.neg
}

class WebSearchGene(Gene):
    """Gen de búsqueda web que utiliza DuckDuckGo."""
    def __init__(self, query_var: str = "query", output_key: str = "web_summary", sentences: int = 3):
        self.query_var = query_var
        self.output_key = output_key
        self.sentences = sentences

    def _search(self, query: str) -> list[str]:
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.select('a.result__a')
            return [link['href'] for link in links if link.has_attr('href')][:3]
        except Exception as e:
            print(f"[WebSearchGene] ERROR en búsqueda: {e}")
            return []

    def _scrape(self, url: str) -> str | None:
        try:
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text() for p in paragraphs)
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            print(f"[WebSearchGene] ERROR en scrapeo de {url[:50]}...: {e}")
            return None

    def execute(self, context: ExecutionContext):
        query = context.get(self.query_var)
        if not query:
            context.set(self.output_key, None)
            return
        urls = self._search(query)
        if not urls:
            context.set(self.output_key, None)
            return
        
        texts = [text for url in urls if (text := self._scrape(url))]
        if texts:
            full_text = " ".join(texts)
            parser = PlaintextParser.from_string(full_text, Tokenizer("spanish"))
            summarizer = LsaSummarizer()
            summary = " ".join(str(s) for s in summarizer(parser.document, self.sentences))
            context.set(self.output_key, summary)
        else:
            context.set(self.output_key, None)

class ScientificSearchGene(Gene):
    """Busca artículos en bases de datos científicas como PubMed."""
    def __init__(self, query_var: str = "main_topic", output_key: str = "scientific_summary", max_results: int = 3):
        self.query_var = query_var
        self.output_key = output_key
        self.max_results = max_results
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    def _search_pubmed(self, query: str) -> list[str]:
        params = {'db': 'pubmed', 'term': query, 'retmax': self.max_results, 'retmode': 'json'}
        try:
            response = requests.get(f"{self.base_url}esearch.fcgi", params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('esearchresult', {}).get('idlist', [])
        except Exception as e:
            print(f"[ScientificSearchGene] ERROR en búsqueda: {e}")
            return []

    def _fetch_abstracts(self, ids: list[str]) -> str:
        if not ids: return ""
        params = {'db': 'pubmed', 'id': ",".join(ids), 'retmode': 'xml', 'rettype': 'abstract'}
        try:
            response = requests.get(f"{self.base_url}efetch.fcgi", params=params, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'xml')
            abstracts = soup.find_all('AbstractText')
            return " ".join([abstract.get_text(strip=True) for abstract in abstracts])
        except Exception as e:
            print(f"[ScientificSearchGene] ERROR en fetch: {e}")
            return ""

    def execute(self, context: ExecutionContext):
        query = context.get(self.query_var)
        if not query:
            context.set(self.output_key, "No se especificó un tema.")
            return
        article_ids = self._search_pubmed(query)
        if not article_ids:
            context.set(self.output_key, f"No encontré artículos en PubMed para '{query}'.")
            return
        context.set(self.output_key, self._fetch_abstracts(article_ids))


class CalculatorGene(Gene):
    """Resuelve expresiones matemáticas de forma segura usando AST."""
    def _eval(self, node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            return SAFE_OPERATORS[type(node.op)](self._eval(node.left), self._eval(node.right))
        elif isinstance(node, ast.UnaryOp):
            return SAFE_OPERATORS[type(node.op)](self._eval(node.operand))
        else:
            raise TypeError(node)

    def execute(self, context: ExecutionContext):
        query = context.get("query", "")
        normalized_query = query.lower().replace(" por ", "*").replace(" más ", "+").replace(" mas ", "+").replace(" menos ", "-").replace(" dividido por ", "/").replace(" elevado a ", "**")
        match = re.search(r'([\d\.\+\-\*\/\(\)\s\^]+)', normalized_query)
        if not match:
            context.set("calculation_result", "No encontré una operación matemática válida.")
            return
        expression = match.group(1).strip()
        try:
            result = self._eval(ast.parse(expression, mode='eval').body)
            context.set("calculation_result", result)
        except Exception as e:
            context.set("calculation_result", f"No pude resolver la expresión. Error: {e}")