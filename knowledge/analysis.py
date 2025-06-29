# prometheus_agi/knowledge/analysis.py
import ast
from collections import Counter

class CodeVisitor(ast.NodeVisitor):
    """Recorre el AST de un archivo para extraer nodos y aristas."""
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
        for base in node.bases:
            try:
                base_name = ast.unparse(base)
                self._add_edge(class_name, base_name, 'INHERITS_FROM')
            except Exception:
                pass
        self.current_class = class_name
        self.generic_visit(node)
        self.current_class = None

    def visit_FunctionDef(self, node):
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
        self.current_function = function_name
        self.generic_visit(node)
        self.current_function = None

    def visit_Call(self, node):
        if not self.current_function:
            return
        try:
            callee_name = ast.unparse(node.func)
            func_name_part = callee_name.split('.')[-1]
            if func_name_part and func_name_part[0].isupper():
                self._add_edge(self.current_function, callee_name, 'CREATES_INSTANCE')
            else:
                self._add_edge(self.current_function, callee_name, 'CALLS')
        except Exception:
            pass

    def visit_Import(self, node):
        for alias in node.names:
            self._add_edge(self.module_name, alias.name, 'IMPORTS')

    def visit_ImportFrom(self, node):
        module = node.module or 'built-in'
        self._add_edge(self.module_name, module, 'IMPORTS_FROM')

class SyntaxVisitor(ast.NodeVisitor):
    """Visitante de AST para extraer información semántica de un Gen."""
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
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            try:
                return f"{self._get_full_call_name(node.func.value)}.{node.func.attr}"
            except:
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
        if full_name.endswith("context.get") and node.args and isinstance(node.args[0], ast.Constant):
            self.analysis["context_gets"].add(node.args[0].value)
        if full_name.endswith("context.set") and node.args and isinstance(node.args[0], ast.Constant):
            self.analysis["context_sets"].add(node.args[0].value)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        self.analysis["loops"] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        self.analysis["conditionals"] += 1
        self.generic_visit(node)