# prometheus_agi/cognition/goal_manager.py
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

# Importaciones del proyecto
from core.base import Goal, GoalStatus

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI

class GoalManager:
    """Gestiona la jerarquía de objetivos de la IA."""
    # Se cambia la anotación de tipo a un string
    def __init__(self, prometheus_mind: 'PrometheusAGI'):
        self.goals: Dict[str, Goal] = {}
        self.prometheus = prometheus_mind

    def add_goal(self, description: str, target: Any, priority: int = 10, parent_id: Optional[str] = None) -> Goal:
        """Añade un nuevo objetivo."""
        new_goal = Goal(description, target, priority=priority, parent_goal_id=parent_id)
        self.goals[new_goal.id] = new_goal
        if parent_id and parent_id in self.goals:
            self.goals[parent_id].sub_goals.append(new_goal)
        print(f"[GOAL_MANAGER] Nuevo objetivo añadido: {new_goal}")
        return new_goal

    def get_active_goal(self) -> Optional[Goal]:
        """Encuentra el objetivo activo de mayor prioridad."""
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        if not active_goals:
            return None
        return max(active_goals, key=lambda g: g.priority)

    def set_goal_status(self, goal_id: str, status: GoalStatus):
        """Cambia el estado de un objetivo."""
        if goal_id in self.goals:
            self.goals[goal_id].status = status
            print(f"[GOAL_MANAGER] Estado del objetivo {goal_id[:4]}... cambiado a {status.value}")

    def update_goals_status(self, current_observation: Dict) -> bool:
        """Comprueba y actualiza el estado de los objetivos basado en la observación."""
        active_goal = self.get_active_goal()
        if not active_goal:
            return False
        
        # Comprueba si el objetivo se ha cumplido
        if isinstance(active_goal.target, np.ndarray) and np.array_equal(current_observation.get("agent"), active_goal.target):
            self.set_goal_status(active_goal.id, GoalStatus.COMPLETED)
            self.prometheus.episodic_memory.record_event(
                event_type="GOAL_COMPLETED",
                details={"description": active_goal.description}
            )
            print(f"[GOAL_MANAGER] ¡OBJETIVO COMPLETADO!: {active_goal.description}")
            return True
        return False