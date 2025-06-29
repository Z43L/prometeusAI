# prometheus_agi/environments/grid_world.py
import time
import gymnasium as gym
import numpy as np
from multiprocessing import Process, Queue, Event
import os
# Importaciones del proyecto
from core.base import Percept

class GridWorldEnv(gym.Env):
    """Entorno simple de mundo en cuadrícula (GridWorld)."""
    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(self, size=10, render_mode='ansi'):
        super().__init__()
        self.size = size
        self.action_space = gym.spaces.Discrete(4) # 0:Up, 1:Down, 2:Left, 3:Right
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "goal": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.goal_pos = self.agent_pos
        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        return self._get_obs(), self._get_info()

    def step(self, action):
        direction_map = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        move = direction_map.get(action, [0, 0])
        prev_distance = self._get_info()["distance"]
        self.agent_pos = np.clip(self.agent_pos + move, 0, self.size - 1)
        
        terminated = np.array_equal(self.agent_pos, self.goal_pos)
        if terminated: reward = 100.0
        elif self._get_info()["distance"] < prev_distance: reward = 5.0
        else: reward = -1.0
            
        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self):
        return {"agent": self.agent_pos, "goal": self.goal_pos}

    def _get_info(self):
        return {"distance": np.abs(self.agent_pos - self.goal_pos).sum()}

    def render(self):
        grid = np.full((self.size, self.size), "_", dtype=str)
        grid[tuple(self.agent_pos)] = "P"
        grid[tuple(self.goal_pos)] = "G"
        return "\n".join(" ".join(row) for row in grid)

class WorldStream(Process):
    """Proceso que ejecuta el entorno y emite percepciones."""
    def __init__(self, environment: gym.Env):
        super().__init__()
        self.env = environment
        self.perception_queue = Queue()
        self.action_queue = Queue()
        self.stop_event = Event()

    def run(self):
        print(f"[WORLD_STREAM pid={os.getpid()}] El mundo ha cobrado vida.")
        obs, info = self.env.reset()
        self.perception_queue.put(Percept("observation", {"obs": obs, "info": info, "reward": 0, "terminated": False}))

        while not self.stop_event.is_set():
            action = self.action_queue.get() if not self.action_queue.empty() else None
            if action is not None:
                obs, reward, term, trunc, info = self.env.step(action)
                percept_type = "termination" if term else "observation"
                self.perception_queue.put(Percept(percept_type, {"obs": obs, "info": info, "reward": reward, "terminated": term}))
                if term or trunc:
                    obs, info = self.env.reset()
                    self.perception_queue.put(Percept("observation", {"obs": obs, "info": info, "reward": 0, "terminated": False}))
            time.sleep(0.1)

    def stop(self):
        self.stop_event.set()