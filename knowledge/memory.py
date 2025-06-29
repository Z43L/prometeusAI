# prometheus_agi/knowledge/memory.py
import time
import numpy as np
from typing import List, Dict

class EpisodicMemory:
    """Almacena una secuencia de eventos de la "vida" de la IA."""
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.log: List[Dict] = []
        self.last_episode_id = 0

    def record_event(self, *, event_type: str, details: Dict):
        """Registra un evento con una marca de tiempo y un ID."""
        if len(self.log) >= self.capacity:
            self.log.pop(0)
        self.last_episode_id += 1
        event_record = {
            "id": self.last_episode_id,
            "timestamp": time.time(),
            "type": event_type,
            "details": details
        }
        self.log.append(event_record)

    def get_events_by_type(self, event_type: str, limit: int = 10) -> List[Dict]:
        """Recupera los últimos N eventos de un tipo específico."""
        return [event for event in reversed(self.log) if event["type"] == event_type][:limit]

    def query_similar_situations(self, current_obs: Dict, top_k: int = 1) -> List[Dict]:
        """Encuentra eventos pasados con estado del agente similar."""
        if not self.log or 'agent' not in current_obs:
            return []
        current_agent_pos = np.array(current_obs['agent'])
        action_events = [event for event in self.log if event['type'] == 'ACTION_TAKEN' and 'previous_observation' in event['details']]
        if not action_events:
            return []
            
        distances = []
        for event in action_events:
            if 'agent' in event['details']['previous_observation']:
                past_agent_pos = np.array(event['details']['previous_observation']['agent'])
                distance = np.linalg.norm(current_agent_pos - past_agent_pos)
                distances.append((distance, event))

        if not distances:
            return []
        distances.sort(key=lambda x: x[0])
        return [event for _, event in distances[:top_k]]

    def get_last_failed_goal_context(self) -> List[Dict]:
        """Recupera la secuencia de acciones que precedieron al último fallo."""
        try:
            last_failure_index = next(i for i, event in enumerate(reversed(self.log)) if event["type"] == "GOAL_FAILED")
            last_failure_index = len(self.log) - 1 - last_failure_index
            start_index = 0
            for i in range(last_failure_index - 1, -1, -1):
                if self.log[i]["type"] in ["GOAL_SET", "EPISODE_START"]:
                    start_index = i
                    break
            return self.log[start_index : last_failure_index + 1]
        except StopIteration:
            return []