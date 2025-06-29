# prometheus_agi/core/base.py
import abc
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

# --- Clases Base ---

class Gene(abc.ABC):
    """Clase base abstracta para todos los genes."""
    @abc.abstractmethod
    def execute(self, context: 'ExecutionContext'):
        raise NotImplementedError

@dataclass
class ExecutionContext:
    """Almacena el estado de la memoria durante la ejecución de un cromosoma."""
    memory: Dict[str, Any] = field(default_factory=dict)
    final_answer_text: str = ""
    thought_log: List[str] = field(default_factory=list)

    def set(self, key: str, value: Any):
        if not isinstance(self.memory, dict):
            print(f"[ERROR_CONTEXT] Context memory corrupted. Type became {type(self.memory)}")
            return
        self.memory[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if not isinstance(self.memory, dict):
            print(f"[ERROR_CONTEXT] Context memory corrupted. Type became {type(self.memory)}")
            return default
        return self.memory.get(key, default)

    def set_final_response(self, text: str):
        self.final_answer_text = str(text) if text is not None else ""

    def get_final_response(self) -> str:
        return self.final_answer_text

    def log_thought(self, component: str, reasoning: str):
        self.thought_log.append(f"[{component}]: {reasoning}")

class Chromosome:
    """Representa una secuencia de genes (una estrategia)."""
    def __init__(self, genes: List[Gene], description: str):
        self.genes = genes
        self.description = description
        self.fitness: float = 0.0
        self.final_context: Optional[ExecutionContext] = None

    def __repr__(self):
        return f"<Chromosome ('{self.description}') | Fitness: {self.fitness:.2f}>"

# --- Clases de Estado y Percepción ---

class GoalStatus(Enum):
    """Define los posibles estados de un objetivo."""
    INACTIVE = "Inactivo"
    ACTIVE = "Activo"
    PAUSED = "Pausado"
    COMPLETED = "Completado"
    FAILED = "Fallido"

@dataclass
class Goal:
    """Representa una ambición o meta."""
    description: str
    target: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: GoalStatus = GoalStatus.ACTIVE
    priority: int = 10
    creation_timestamp: float = field(default_factory=time.time)
    parent_goal_id: Optional[str] = None
    sub_goals: List['Goal'] = field(default_factory=list)

    def __repr__(self):
        return f"<Goal(id={self.id[:4]}..., desc='{self.description}', status={self.status.value})>"

@dataclass
class Percept:
    """Representa una unidad de percepción del entorno."""
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)

# --- Descriptores y Clases Compuestas ---

@dataclass
class GeneDescriptor:
    """Metadatos que describen la capacidad de un Gen para el StrategicPlanner."""
    gene_class: type
    purpose_description: str
    relevant_intents: List[str] = field(default_factory=list)
    input_variables: List[str] = field(default_factory=list)
    output_variable: Optional[str] = None
    is_terminal: bool = False
    priority: int = 10

class CompositeGene(Gene):
    """Un gen compuesto por una secuencia de otros genes."""
    def __init__(self, name: str, sequence: List[Gene]):
        self.name = name
        self.sequence = sequence

    def execute(self, context: ExecutionContext):
        for gene in self.sequence:
            gene.execute(context)
            if context.get("execution_error"):
                break

    def __repr__(self):
        return f"<CompositeGene({self.name})>"