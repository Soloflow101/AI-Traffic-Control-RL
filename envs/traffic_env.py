import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Tuple, Dict

class TrafficEnv(gym.Env):
    def __init__(self, simulator):
        super(TrafficEnv, self).__init__()
        self.sim = simulator
        
        cfg = self.sim.config
        # Look inside the 'env' section of your config.yaml
        env_cfg = cfg.get("env", {})
        self.min_green = env_cfg.get("min_green", 5)
        
        self.current_phase_time = 0
        self.current_phase = 0

        self.action_space = spaces.Discrete(2)

        # UPDATED: 7 dimensions [Q_N, Q_S, Q_E, Q_W, P_NS, P_EW, Phase]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(7,), dtype=np.float32
        )

    def _get_obs(self, sim_state: Dict) -> np.ndarray:
        """Picks the 7 values from the simulator to give to the AI."""
        return np.array([
            sim_state["q_n"],
            sim_state["q_s"],
            sim_state["q_e"],
            sim_state["q_w"],
            sim_state["p_ns"],
            sim_state["p_ew"],
            sim_state["phase"]
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        sim_state = self.sim.reset()
        self.current_phase = sim_state.get("phase", 0)
        self.current_phase_time = 0
        
        obs = self._get_obs(sim_state)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Logic: If AI wants to switch (1) but min_green isn't met, force stay (0)
        actual_action = action
        if action == 1 and self.current_phase_time < self.min_green:
            actual_action = 0

        sim_state, reward, done, info = self.sim.step(actual_action)

        # Update timer based on whether the phase actually changed
        if sim_state.get("phase", 0) != self.current_phase:
            self.current_phase = sim_state.get("phase", 0)
            self.current_phase_time = 0
        else:
            self.current_phase_time += self.sim.time_step

        obs = self._get_obs(sim_state)
        
        # Return Gymnasium standard: obs, reward, terminated, truncated, info
        return obs, float(reward), done, False, info

    def render(self):
        pass

    def close(self):
        self.sim.close()
