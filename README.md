# AI Traffic-Light Regulation Model

A single-intersection traffic signal control demo that optimizes the order of traffic lights (green/red phases) and pedestrian crossing to minimize vehicle delay and pedestrian wait time.

## Overview

This project scaffolds a traffic signal control system using:
- **Simulator**: MockSimulator (lightweight, no SUMO required) or SUMO TraCI (extended version)
- **Control strategies**: Fixed-time, Random, and RL-based (placeholder for PPO/DQN)
- **Evaluation metrics**: Average queue length, vehicle delay, pedestrian wait time, fairness
- **Tech stack**: Python, NumPy, Stable-Baselines3 (optional)

## Quick Start (Windows)

```powershell
# Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies (minimal for mock sim; add sumolib/traci if using SUMO)
pip install -r requirements.txt
```

## Run Evaluation (MockSimulator, no SUMO)

```powershell
python eval\evaluate.py
```

Output:
- Fixed-time baseline average queue length
- Random policy average queue length
- Compare performance

## Run Training Loop

```powershell
python agents\train_rl.py
```

(Currently runs a random policy; placeholder for PPO training.)

## Project Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── configs/
│   └── config.yaml                     # Scenario & reward config
├── simulators/
│   └── simulator_interface.py           # MockSimulator class
├── envs/
│   └── traffic_env.py                  # Gym-like environment wrapper
├── baselines/
│   ├── fixed_time.py                   # Fixed-time signal controller
│   └── actuated.py                     # Detector-based actuated (stub)
├── agents/
│   ├── train_rl.py                     # Training loop (RL placeholder)
│   └── models.py                       # NN policy/value (stub)
├── eval/
│   └── evaluate.py                     # Run baselines & compute metrics
└── run_train.bat, run_eval.bat         # Windows convenience scripts
```

## Key Components

### MockSimulator
Simulates a single intersection with two approaches (North-South, East-West).
- **State**: vehicle queue lengths (NS, EW), pedestrian queue lengths (NS, EW), current phase
- **Actions**: keep phase (0) or request phase switch (1)
- **Dynamics**: vehicles/pedestrians arrive randomly; serve based on green phase
- **Reward**: negative sum of queue lengths

### TrafficEnv
Gym-like environment wrapper with min-green enforcement and phase-timing logic.
- `reset()` – reset sim and phase timer
- `step(action)` – apply action, advance simulation, return observation + reward
- Enforces **min-green** constraint to avoid unphysical rapid switching

### Fixed-Time Controller
Simple baseline that alternates phases on a fixed schedule (Webster default).

### Evaluation
Runs multiple episodes for each controller and computes average metrics:
- **Average Queue Length** (lower is better)
- Extensible for vehicle delay, travel time, emissions

## Next Steps

### 1. SUMO Integration (Optional)
- Install SUMO: [https://eclipse.dev/sumo/](https://eclipse.dev/sumo/)
- Create a SUMO network file (`.net.xml`)
- Extend `simulator_interface.py` with `SUMOSimulator` class
- Point `config.yaml` to your `.net.xml` and set `simulator: sumo`

### 2. RL Agent Training
- Install Stable-Baselines3: `pip install stable-baselines3`
- Extend `agents/train_rl.py` to use `PPO` or `DQN` from SB3
- Define action space (discrete: phase switch; continuous: green time)
- Log rewards and policy performance to TensorBoard

### 3. Multi-Objective Reward
- Adjust reward weights in `config.yaml` (alpha_vehicle, beta_pedestrian)
- Implement fairness metrics: ensure both directions and pedestrians are served fairly
- Add constraints: max queue length, min pedestrian crossing time

### 4. Evaluation & Visualization
- Extend `eval/evaluate.py` to plot queue dynamics and compare controllers
- Add metrics: vehicle delay, pedestrian wait, throughput, emissions
- Run ablation studies: test robustness to demand shifts, sensor noise

### 5. Corridor / Network Extension
- Scale from single intersection to 2-5 intersections
- Implement coordination: share observations between intersections
- Use multi-agent RL (MARL) libraries like Ray RLlib

## Notes

- **Mock simulator**: runs instantly; good for rapid prototyping
- **SUMO**: adds fidelity but requires network setup and is slower
- **Safety**: All-real (all-red), min-green, and phase veto logic are intentionally simple; real systems require formal verification
- **Sim-to-Real**: sensor noise, pedestrian behavior, and unusual events require robust transfer strategies

## Files of Interest

- **Baseline comparison**: [baselines/fixed_time.py](baselines/fixed_time.py)
- **Environment definition**: [envs/traffic_env.py](envs/traffic_env.py)
- **Evaluation harness**: [eval/evaluate.py](eval/evaluate.py)
- **Config**: [configs/config.yaml](configs/config.yaml)

## References

- SUMO Traffic Simulator: https://eclipse.dev/sumo/
- Traffic Signal Control via RL: https://traffic-signal-control.github.io/
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/

