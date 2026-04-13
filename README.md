# 🚦 RL Traffic Signal Controller

A **Reinforcement Learning** based traffic signal controller that uses **Q-Learning** to intelligently manage a 4-way intersection. The agent learns to minimize vehicle waiting time, prevent lane starvation, and ensure fair signal distribution — all visualized in real-time using **Pygame**.

---

## 📌 Project Overview

Traditional traffic signals use fixed timers that ignore real-time traffic conditions. This project replaces that with a Q-Learning agent that:

- Observes the current traffic state (vehicle counts + wait times per lane)
- Decides which lane gets the green signal
- Learns from rewards to improve decisions over time
- Runs live inside a Pygame traffic simulation

---

## 🧠 How It Works

```
Traffic Simulation (Pygame)
        ↓
   RLBridge — reads vehicle counts & wait times
        ↓
   QLearningAgent — picks best action (green direction)
        ↓
   TrafficEnv — computes reward
        ↓
   Q-Table updated → agent improves over time
```

### State
The agent observes **8 values** — for each of the 4 directions:
- Queue bucket (0–5): how many cars are waiting
- Wait bin (0–3): how long the lane has been waiting

### Actions
The agent picks **1 of 4 directions** to give green:
- `0` → Right
- `1` → Down
- `2` → Left
- `3` → Up

### Reward Function
| Component | Description |
|---|---|
| ✅ Throughput bonus | +3.0 per car cleared |
| ❌ Queue penalty | -1.0 per waiting car |
| ❌ Starvation penalty | -8.0 per step a lane waits beyond limit |
| ✅ Urgency bonus | +10.0 for serving a starving lane |
| ❌ Monopoly penalty | -5.0 if same lane gets green 3× in a row |
| ❌ Fairness penalty | -0.5 × queue imbalance across lanes |

---

## 📁 Project Structure

```
RL-Traffic-Signal-Controller/
│
├── rl_agent.py          # Q-Learning agent (epsilon-greedy, Q-table, save/load)
├── rl_bridge.py         # Bridge between Pygame simulation and RL agent
├── traffic_env.py       # Custom traffic environment (state, step, reward)
├── train.py             # Headless training script (no Pygame window)
├── simulation.py        # Full Pygame simulation with live RL HUD
│
├── images/
│   ├── intersection.png
│   ├── signals/
│   │   ├── red.png
│   │   ├── yellow.png
│   │   └── green.png
│   ├── right/           # car, bus, truck, bike sprites
│   ├── down/
│   ├── left/
│   └── up/
│
├── qtable.json          # Saved Q-table (generated after training)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/MahendraBudda/RL-Traffic-Signal-Controller.git
cd RL-Traffic-Signal-Controller
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1 — Train the agent (headless, no window)
```bash
python train.py
```
Optional arguments:
```bash
python train.py --episodes 3000 --steps 300
```
This saves the Q-table to `qtable.json` and prints progress every 100 episodes.

### Step 2 — Run the Pygame simulation
```bash
python simulation.py
```
The agent loads the trained Q-table and controls the intersection live. Online learning continues during the simulation.

---

## 🖥️ Simulation HUD

The simulation displays a real-time HUD overlay showing:

| Panel | Info |
|---|---|
| **Vehicles Crossed** | Count per vehicle type + total |
| **Direction Counts** | Live waiting vehicles per lane (color-coded) |
| **RL Agent HUD** | Current state, Q-values, wait bars, explore/exploit mode, reward |

- 🟢 Green bar → low wait
- 🟡 Yellow bar → moderate wait
- 🔴 Red bar → lane is starving

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Algorithm | Q-Learning |
| Learning rate (α) | 0.1 |
| Discount factor (γ) | 0.9 |
| Epsilon start | 1.0 |
| Epsilon min | 0.05 |
| Epsilon decay | 0.995 per episode |
| Steps per episode | 300 |
| State space | 8-tuple (queue bucket × 4 + wait bin × 4) |
| Action space | 4 (one per direction) |

---

## 🔧 Key Design Decisions

### ✅ Unified State + Reward
Training (`train.py`) and simulation (`rl_bridge.py`) use **identical** state encoding and reward function — so the agent transfers perfectly from headless training to live simulation.

### ✅ Starvation Prevention
If any lane waits beyond **2× the starvation limit**, the bridge forces it green. This acts as a safety override while the agent is still learning.

### ✅ Online Learning
The agent continues to learn **during the simulation** — `feedback()` is called every tick, not just once per signal cycle, so the agent learns cause-effect relationships quickly.

### ✅ Q-Table Persistence
The Q-table is saved to `qtable.json` periodically during training and on simulation exit — so progress is never lost between runs.

---

## 📦 Requirements

```
numpy>=1.24.0
pygame>=2.5.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## 🗺️ Future Improvements

- [ ] Replace Q-table with Deep Q-Network (DQN) for larger state spaces
- [ ] Add multi-intersection support
- [ ] Pedestrian crossing signals
- [ ] Real traffic data integration
- [ ] Performance graphs and training analytics

---

## 👤 Author

**Mahendra Budda**
- GitHub: [@MahendraBudda](https://github.com/MahendraBudda)

