"""
traffic_env.py
==============
Unified Traffic Environment used by BOTH training and simulation.
Fixes applied:
  - Reward computed BEFORE queue update (correct released count)
  - Consistent state: (counts, wait_bins) everywhere
  - Identical reward function in training and bridge
  - Starvation + fairness + throughput rewards unified
  - Monopoly detection
"""

import random


class TrafficEnv:

    DIRECTIONS = ["right", "down", "left", "up"]

    # Starvation config
    STARVATION_LIMIT   = 15   # wait-steps before a lane is "starving"
    STARVATION_PENALTY = 8.0  # extra penalty per step beyond limit
    MONOPOLY_PENALTY   = 5.0  # penalty for giving same lane green 3× in a row

    def __init__(
        self,
        arrival_probs=None,
        max_cars_per_green=2,
        max_steps=300,
        max_queue=20,
    ):
        self.arrival_probs      = arrival_probs or [0.35, 0.35, 0.35, 0.35]
        self.max_cars_per_green = max_cars_per_green
        self.max_steps          = max_steps
        self.max_queue          = max_queue
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self):
        self.queues       = [0, 0, 0, 0]
        self.wait_steps   = [0, 0, 0, 0]
        self.step_num     = 0
        self.total_wait   = 0
        self.last_actions = []          # for monopoly detection (last 3)
        return self.get_state()

    # ----------------------------------------------------------------- state
    def get_state(self):
        counts    = {d: self.queues[i]               for i, d in enumerate(self.DIRECTIONS)}
        wait_bins = {d: self._wait_bin(self.wait_steps[i]) for i, d in enumerate(self.DIRECTIONS)}
        return counts, wait_bins

    def _wait_bin(self, steps):
        """Bin waiting time into 4 coarse levels (0-3)."""
        if steps <= 3:
            return 0
        elif steps <= 8:
            return 1
        elif steps <= self.STARVATION_LIMIT:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------ step
    def step(self, action):
        # 1. Random arrivals
        for i in range(4):
            if random.random() < self.arrival_probs[i]:
                self.queues[i] = min(self.queues[i] + 1, self.max_queue)

        # 2. How many cars will be released (BEFORE modifying queue)
        released = min(self.queues[action], self.max_cars_per_green)

        # 3. Compute reward BEFORE updating queue  ← FIX #1
        reward = self._compute_reward(action, released)

        # 4. Apply release
        self.queues[action] -= released

        # 5. Update wait timers
        for i in range(4):
            if i == action:
                self.wait_steps[i] = 0
            else:
                self.wait_steps[i] += 1

        # 6. Track total wait
        self.total_wait += sum(self.queues)
        self.step_num   += 1

        # 7. Track recent actions for monopoly detection
        self.last_actions.append(action)
        if len(self.last_actions) > 3:
            self.last_actions.pop(0)

        done          = self.step_num >= self.max_steps
        counts, wait_bins = self.get_state()

        info = {
            "step"         : self.step_num,
            "queues"       : list(self.queues),
            "wait_steps"   : list(self.wait_steps),
            "total_waiting": sum(self.queues),
            "green_dir"    : self.DIRECTIONS[action],
        }
        return counts, wait_bins, reward, done, info

    # -------------------------------------------------------------- reward
    def _compute_reward(self, action, released):
        """
        Unified reward function — identical logic is reproduced in RLBridge
        so that online learning and offline training receive the same signal.

        Parameters
        ----------
        action   : int  (0-3, the chosen green direction)
        released : int  (cars actually cleared this step — computed BEFORE queue update)
        """
        reward = 0.0

        # A. Throughput bonus
        reward += released * 3.0

        # B. Queue penalty (current queues BEFORE release, so still accurate)
        total_queue = sum(self.queues)
        reward     -= total_queue * 1.0

        # C. Starvation penalty — hard penalty for lanes that waited too long
        for i in range(4):
            if i != action and self.wait_steps[i] > self.STARVATION_LIMIT:
                overshoot = self.wait_steps[i] - self.STARVATION_LIMIT
                reward   -= self.STARVATION_PENALTY * overshoot

        # D. Starvation urgency — bonus for finally serving a starving lane
        if self.wait_steps[action] > self.STARVATION_LIMIT:
            reward += 10.0

        # E. Monopoly penalty
        if len(self.last_actions) == 3 and len(set(self.last_actions)) == 1:
            reward -= self.MONOPOLY_PENALTY

        # F. Fairness — penalise imbalanced queues
        if max(self.queues) > 0:
            imbalance = max(self.queues) - min(self.queues)
            reward   -= imbalance * 0.5

        return reward

    # -------------------------------------------------------- public helpers
    def vehicle_counts(self):
        return {d: self.queues[i] for i, d in enumerate(self.DIRECTIONS)}

    def wait_bins(self):
        return {d: self._wait_bin(self.wait_steps[i]) for i, d in enumerate(self.DIRECTIONS)}

    def compute_reward_from_counts(self, action, counts, wait_steps, released):
        """
        Stateless version of the reward function — used by RLBridge so the
        online reward signal is IDENTICAL to the one used during training.

        Parameters
        ----------
        action     : int
        counts     : dict {direction: int}  vehicle counts BEFORE release
        wait_steps : dict {direction: int}  raw wait counters
        released   : int                    cars cleared this step
        """
        dirs = self.DIRECTIONS
        queues_list = [counts.get(d, 0) for d in dirs]
        waits_list  = [wait_steps.get(d, 0) for d in dirs]

        reward = 0.0

        reward += released * 3.0

        total_queue = sum(queues_list)
        reward     -= total_queue * 1.0

        for i, d in enumerate(dirs):
            if i != action and waits_list[i] > self.STARVATION_LIMIT:
                overshoot = waits_list[i] - self.STARVATION_LIMIT
                reward   -= self.STARVATION_PENALTY * overshoot

        if waits_list[action] > self.STARVATION_LIMIT:
            reward += 10.0

        # Note: monopoly tracking lives in RLBridge for the simulation path
        if max(queues_list) > 0:
            imbalance = max(queues_list) - min(queues_list)
            reward   -= imbalance * 0.5

        return reward