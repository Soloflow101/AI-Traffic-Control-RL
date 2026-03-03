import numpy as np
from stable_baselines3 import PPO
from envs.traffic_env import TrafficEnv
from simulators.simulator_interface import MockSimulator
import yaml

# 1. Load Model and Env
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

sim = MockSimulator(config)
env = TrafficEnv(sim)
model = PPO.load("models/ppo_traffic_agent.zip")

# 2. Run a Test Episode
obs, _ = env.reset()
total_reward = 0
for _ in range(3600): # Run for 1 hour of simulated time
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done: break

print(f"AI Total Reward (Efficiency): {total_reward}")
