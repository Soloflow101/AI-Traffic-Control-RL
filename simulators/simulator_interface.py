from typing import Dict
import random
import numpy as np

class MockSimulator:
    def __init__(self, config: Dict):
        self.config = config
        self.episode_length = config.get('episode_length', 3600)
        self.time_step = config.get('time_step', 1)
        self.time = 0
        
        # Arrival rates from config
        self.v_rate = config.get('arrival_rate', 1.2)
        self.p_rate = config.get('ped_arrival_rate', 0.6)
        
        # Reward weights
        reward_cfg = config.get('reward', {})
        self.alpha = reward_cfg.get('alpha_vehicle', 1.0)
        self.beta = reward_cfg.get('beta_pedestrian', 3.0)

        self.state = {
            "q_n": 0, "q_s": 0, "q_e": 0, "q_w": 0,
            "p_ns": 0, "p_ew": 0,
            "phase": 0 # 0: NS Green, 1: EW Green
        }

    def reset(self):
        self.time = 0
        for key in self.state:
            if key != "phase":
                self.state[key] = random.randint(0, 3)
        return self.get_state()

    def get_state(self) -> Dict:
        return dict(self.state)

    def _compute_reward(self) -> float:
        # 1. Base Wait Penalty
        v_wait = self.state["q_n"] + self.state["q_s"] + self.state["q_e"] + self.state["q_w"]
        p_wait = self.state["p_ns"] + self.state["p_ew"]
        penalty = (v_wait * self.alpha) + (p_wait * self.beta)
        
        # 2. HARSH Idle Green Penalty (Ensures AI snaps lights off)
        idle_penalty = 0
        if self.state["phase"] == 0 and (self.state["q_n"] + self.state["q_s"] == 0):
            idle_penalty = 35.0 
        if self.state["phase"] == 1 and (self.state["q_e"] + self.state["q_w"] == 0):
            idle_penalty = 35.0

        return -float(penalty + idle_penalty)

    def step(self, action: int):
        # Apply action
        if action == 1:
            self.state["phase"] = 1 - self.state["phase"]

        # 1. Arrivals
        for lane in ["q_n", "q_s", "q_e", "q_w"]:
            self.state[lane] += np.random.poisson(self.v_rate / 2)
        
        self.state["p_ns"] += np.random.poisson(self.p_rate)
        self.state["p_ew"] += np.random.poisson(self.p_rate)

        # 2. Discharge
        discharge = self.config.get('discharge_rate', 4)
        if self.state["phase"] == 0: # NS Green
            self.state["q_n"] = max(0, self.state["q_n"] - random.randint(1, discharge))
            self.state["q_s"] = max(0, self.state["q_s"] - random.randint(1, discharge))
            self.state["p_ew"] = max(0, self.state["p_ew"] - random.randint(1, 2))
        else: # EW Green
            self.state["q_e"] = max(0, self.state["q_e"] - random.randint(1, discharge))
            self.state["q_w"] = max(0, self.state["q_w"] - random.randint(1, discharge))
            self.state["p_ns"] = max(0, self.state["p_ns"] - random.randint(1, 2))

        self.time += self.time_step
        done = self.time >= self.episode_length
        
        return self.get_state(), self._compute_reward(), done, {"time": self.time}

    def close(self):
        pass
