import os
import yaml
import numpy as np
from stable_baselines3 import PPO
from envs.traffic_env import TrafficEnv
from simulators.simulator_interface import MockSimulator

def run_evaluation():
    # 1. Load Settings
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['episode_length'] = 3600

    def run_sim(model=None):
        sim = MockSimulator(config)
        env = TrafficEnv(sim)
        obs, _ = env.reset()
        
        metrics = {"veh_wait": 0, "ped_wait": 0}
        
        for step in range(3600):
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Baseline: Switch every 30 seconds
                action = 1 if (step % 30 == 0 and step > 0) else 0
            
            obs, _, done, _, _ = env.step(action)
            # obs = [Q_N, Q_S, Q_E, Q_W, P_NS, P_EW, Phase]
            metrics["veh_wait"] += sum(obs[0:4]) # Sum all 4 car lanes
            metrics["ped_wait"] += sum(obs[4:6]) # Sum both ped crossings
            if done: break
        return metrics

    # --- EXECUTION ---
    print("🚦 Running Baseline (Fixed Timer)...")
    fixed_results = run_sim(model=None)

    print("🧠 Running AI Agent...")
    model_path = "models/ppo_traffic_agent.zip"
    if not os.path.exists(model_path):
        print("❌ Error: Model not found!")
        return
        
    model = PPO.load(model_path)
    ai_results = run_sim(model=model)

    # --- RESULTS ---
    print("\n" + "="*45)
    print("⚖️  4-LANE TRAFFIC EVALUATION (1 HOUR)")
    print("="*45)
    
    v_timer = fixed_results["veh_wait"]/3600
    v_ai = ai_results["veh_wait"]/3600
    v_imp = ((v_timer - v_ai) / v_timer) * 100
    print(f"🚗 VEHICLES   | Timer: {v_timer:.1f} | AI: {v_ai:.1f} | Δ: {v_imp:.1f}%")
    
    p_timer = fixed_results["ped_wait"]/3600
    p_ai = ai_results["ped_wait"]/3600
    p_imp = ((p_timer - p_ai) / p_timer) * 100
    print(f"🚶 PEDESTRIANS | Timer: {p_timer:.1f} | AI: {p_ai:.1f} | Δ: {p_imp:.1f}%")
    
    print("-" * 45)
    print(f"🚀 Overall AI Performance: {(v_imp + p_imp)/2:.1f}% Improvement")
    print("="*45)

if __name__ == "__main__":
    run_evaluation()
