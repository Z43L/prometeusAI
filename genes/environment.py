# prometheus_agi/genes/environment.py
import random
import numpy as np
from typing import TYPE_CHECKING

# Importaciones del proyecto
from core.base import Gene, ExecutionContext

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from core.mind import PrometheusAGI
    from cognition.goal_manager import GoalManager

class PerceiveEnvironmentGene(Gene):
    """Procesa la observación del entorno y la guarda en el contexto."""
    def __init__(self, observation_var: str = "observation"):
        self.observation_var = observation_var

    def execute(self, context: ExecutionContext):
        observation = context.get(self.observation_var)
        if isinstance(observation, dict):
            agent_pos = observation.get("agent")
            goal_pos = observation.get("goal")
            if agent_pos is not None and goal_pos is not None:
                context.set("agent_pos", tuple(agent_pos))
                context.set("goal_pos", tuple(goal_pos))
                context.set("perceived_world", f"Estoy en {tuple(agent_pos)}, meta en {tuple(goal_pos)}.")

class DecideActionGene(Gene):
    """Decide una acción basada en objetivos, o explora aleatoriamente."""
    def __init__(self, output_var: str = "chosen_action"):
        self.output_var = output_var
        self.action_map = {0: "ARRIBA", 1: "ABAJO", 2: "IZQUIERDA", 3: "DERECHA"}

    def execute(self, context: ExecutionContext):
        active_goal = context.get("active_goal")
        agent_pos = context.get("agent_pos")
        action = None

        if active_goal and agent_pos:
            target_pos = np.array(active_goal.target)
            agent_pos_arr = np.array(agent_pos)
            delta = target_pos - agent_pos_arr
            
            if np.array_equal(delta, [0, 0]):
                action = None # Objetivo alcanzado, no hacer nada
            elif abs(delta[0]) > abs(delta[1]):
                action = 1 if delta[0] > 0 else 0  # Movimiento vertical
            else:
                action = 3 if delta[1] > 0 else 2  # Movimiento horizontal
        
        if action is None:
             action = random.choice(list(self.action_map.keys()))
             print(f"  [DecideAction] Sin objetivo, explorando hacia {self.action_map[action]}.")

        context.set(self.output_var, action)

class CheckGoalStatusGene(Gene):
    """Consulta el GoalManager y pone el objetivo activo en el contexto."""
    def __init__(self, goal_manager: 'GoalManager', output_var: str = "active_goal"):
        self.goal_manager = goal_manager
        self.output_var = output_var

    def execute(self, context: ExecutionContext):
        active_goal = self.goal_manager.get_active_goal()
        if active_goal:
            context.set(self.output_var, active_goal)

class CuriosityGene(Gene):
    """Establece un objetivo interno de exploración en el GoalManager."""
    def __init__(self, mind: 'PrometheusAGI'):
        self.mind = mind

    def execute(self, context: ExecutionContext):
        if self.mind.goal_manager.get_active_goal():
            return

        corner = random.choice([[0,0], [0, self.mind.env_size-1], [self.mind.env_size-1, 0], [self.mind.env_size-1, self.mind.env_size-1]])
        target_pos = np.array(corner)
        description = f"Explorar la esquina en {target_pos}"
        self.mind.goal_manager.add_goal(description, target_pos)