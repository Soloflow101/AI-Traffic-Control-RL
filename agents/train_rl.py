import os
import sys
import yaml
import numpy as np
from stable_baselines3 import PPO

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.traffic_env import TrafficEnv
from simulators.simulator_interface import MockSimulator

def train():
    # 1. Ensure models directory exists
    if not os.path.exists("models"):
        os.makedirs("models")

    # 2. Load the EXTREME config
    with open("configs/config.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    # 3. Initialize Sim and Env
    sim = MockSimulator(config_data) 
    env = TrafficEnv(sim)

    # 4. Define the AI Model
    # We increase the 'learning_rate' slightly for faster adaptation to extreme traffic
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, tensorboard_log="./ppo_logs/")

    # 5. Start Stress Test Training
    print(f"\n🔥 CRASH TEST INITIATED: Training on Extreme Traffic ({config_data.get('arrival_rate')} veh/s)")
    print(f"🚶 PEDESTRIAN WEIGHT: {config_data.get('reward', {}).get('beta_pedestrian', 1.0)}x")
    
    # Increased to 30,000 steps for better convergence on difficult traffic
    model.learn(total_timesteps=50000)

    # 6. Save the High-Performance Brain
    model.save("models/ppo_traffic_agent")
    print("\n✅ Training complete! Extreme-traffic model saved.")

if __name__ == "__main__":
    train()
