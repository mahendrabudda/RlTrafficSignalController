"""
rl_agent.py
===========
Q-Learning agent for traffic signal control.

Fixes applied:
  - State encoding unified: (count_bin, wait_bin) × 4 directions  ← FIX #2 / #3
  - Consistent normalization everywhere                            ← FIX #14
  - current_green removed from state (redundant + causes mismatch) ← FIX #15
  - Epsilon decay properly exposed                                 ← FIX #6
  - Q-table save / load (JSON)                                     ← FIX #12
  - get_q_values() helper for HUD                                  
"""

import json
import random
from collections import defaultdict

import numpy as np


# ── tuneable hyper-parameters ─────────────────────────────────────────────────
ALPHA         = 0.1     # learning rate
GAMMA         = 0.9     # discount factor
EPSILON_START = 1.0     # initial exploration rate
EPSILON_MIN   = 0.05    # floor for exploration
EPSILON_DECAY = 0.995   # multiplicative decay per episode

# State discretisation
MAX_QUEUE    = 20       # queue bucket denominator
MAX_WAIT_BIN = 3        # maximum wait bin value (0–3)


class QLearningAgent:

    DIRECTIONS = ["right", "down", "left", "up"]

    def __init__(self, epsilon=EPSILON_START):
        self.alpha         = ALPHA
        self.gamma         = GAMMA
        self.epsilon       = epsilon
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY

        self.q_table = defaultdict(lambda: np.zeros(4))

        # Telemetry (used by RLBridge metrics)
        self.last_state  = None
        self.last_action = None
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.step_count  = 0
        self.was_explore = False

    # ──────────────────────────────────────────────── state encoding (FIX #2/#3)
    def encode_state(self, counts, wait_bins):
        """
        Unified state tuple used by BOTH training and simulation.

        counts    : dict {direction: int}   — raw vehicle counts
        wait_bins : dict {direction: int}   — binned wait level 0-3

        Returns a hashable tuple:
          (q0_norm, w0, q1_norm, w1, q2_norm, w2, q3_norm, w3)
        where q_norm is bucketed into 0-5 (queue / 4, capped at 5).
        """
        state = []
        for d in self.DIRECTIONS:
            q_bucket = min(int(counts.get(d, 0) // 4), 5)   # 0-5 bucket  ← FIX #13
            w_bin    = int(wait_bins.get(d, 0))              # 0-3
            state.append(q_bucket)
            state.append(w_bin)
        return tuple(state)

    # ────────────────────────────────────────────────────────────── action
    def select_action(self, state):
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            self.was_explore = True
            return random.randint(0, 3)
        self.was_explore = False
        return int(np.argmax(self.q_table[state]))

    # ────────────────────────────────────────────────────────────── update
    def update(self, state, action, reward, next_state):
        """Standard Q-learning Bellman update."""
        old_q    = self.q_table[state][action]
        max_next = np.max(self.q_table[next_state])

        self.q_table[state][action] = old_q + self.alpha * (
            reward + self.gamma * max_next - old_q
        )

        self.last_reward   = reward
        self.total_reward += reward
        self.step_count   += 1

    # ────────────────────────────────────────────────── epsilon management
    def decay_epsilon(self):
        """Call once per episode during training."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ────────────────────────────────────────────────────────── Q-value helper
    def get_q_values(self, state):
        if state is None:
            return [0.0, 0.0, 0.0, 0.0]
        return list(self.q_table[state])

    # ────────────────────────────────────────────────────── persistence (FIX #12)
    def save(self, path="qtable.json"):
        serialisable = {
            str(k): v.tolist()
            for k, v in self.q_table.items()
        }
        with open(path, "w") as f:
            json.dump({"epsilon": self.epsilon, "qtable": serialisable}, f)
        print(f"[QLearningAgent] Q-table saved → {path}  "
              f"({len(self.q_table)} states, ε={self.epsilon:.4f})")

    def load(self, path="qtable.json"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.epsilon = float(data.get("epsilon", self.epsilon))
            for k_str, v in data.get("qtable", {}).items():
                key = tuple(int(x) for x in k_str.strip("()").split(",") if x.strip())
                self.q_table[key] = np.array(v)
            print(f"[QLearningAgent] Q-table loaded ← {path}  "
                  f"({len(self.q_table)} states, ε={self.epsilon:.4f})")
        except FileNotFoundError:
            print(f"[QLearningAgent] No checkpoint at '{path}' — starting fresh.")
        except Exception as e:
            print(f"[QLearningAgent] Load error: {e} — starting fresh.")