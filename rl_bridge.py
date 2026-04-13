"""
rl_bridge.py
============
Bridge between the Pygame simulation and the RL agent.

Fixes applied:
  - State encoding identical to training (FIX #2 / #3)
  - Reward function identical to TrafficEnv._compute_reward (FIX #4)
  - Reward computed with BEFORE-state counts so released is accurate (FIX #1)
  - feedback() called every simulation step, not per cycle (FIX #5)
  - Epsilon decay during online learning (FIX #6)
  - online_learning=True by default (FIX #7)
  - Starvation override reduced; incorporated into reward instead (FIX #8)
  - Monopoly tracking lives here for the simulation path
"""

from rl_agent import QLearningAgent
from traffic_env import TrafficEnv   # for stateless reward helper


class RLBridge:

    DIRECTIONS     = ["right", "down", "left", "up"]
    STARVATION_LIM = 15          # wait-steps — matches TrafficEnv

    def __init__(
        self,
        qtable_path="qtable.json",
        online_learning=True,    # FIX #7 — enable by default
        epsilon=0.2,             # FIX #6 — start with some exploration
        save_every=500,          # checkpoint frequency (steps)
        qtable_save_path="qtable.json",
    ):
        self.agent          = QLearningAgent(epsilon=epsilon)
        self.agent.load(qtable_path)
        self.online         = online_learning
        self.save_every     = save_every
        self.save_path      = qtable_save_path

        self._prev_state    = None
        self._prev_action   = None
        self._prev_counts   = None   # counts BEFORE last action (for reward)

        # Wait timers in raw steps (same scale as TrafficEnv.wait_steps)
        self.wait_steps = {d: 0 for d in self.DIRECTIONS}

        # Monopoly detection (last 3 actions)
        self._last_actions  = []

        # Stateless reward helper (reuses TrafficEnv logic)
        self._env_helper    = TrafficEnv()

    # ──────────────────────────────────────────────────────────────── choose
    def choose(self, vehicles):
        """
        Called once per signal cycle to pick the next green direction.
        Returns an action index 0-3.
        """
        counts    = self._count_waiting(vehicles)
        wait_bins = self._get_wait_bins()
        state     = self.agent.encode_state(counts, wait_bins)

        # FIX #8 — only override at 2× the limit AND reduce frequency.
        # Starvation is now mostly handled via reward, not hard overrides.
        forced = self._check_starvation()
        if forced is not None:
            action             = forced
            self.agent.was_explore = False
        else:
            action = self.agent.select_action(state)

        # Save previous counts for reward computation in feedback()
        self._prev_counts  = counts.copy()
        self._prev_state   = state
        self._prev_action  = action

        # Update wait timers
        for i, d in enumerate(self.DIRECTIONS):
            if i == action:
                self.wait_steps[d] = 0
            else:
                self.wait_steps[d] += 1

        # Monopoly tracking
        self._last_actions.append(action)
        if len(self._last_actions) > 3:
            self._last_actions.pop(0)

        return action

    # ──────────────────────────────────────────────────────────── feedback
    def feedback(self, vehicles):
        """
        FIX #5 — call this every simulation step, not once per cycle.
        Computes a reward identical to TrafficEnv and updates the Q-table.
        """
        if not self.online or self._prev_state is None:
            return

        counts    = self._count_waiting(vehicles)
        wait_bins = self._get_wait_bins()
        nxt_state = self.agent.encode_state(counts, wait_bins)

        # --- Compute reward using the SAME formula as TrafficEnv (FIX #4) ---
        action   = self._prev_action
        prev_c   = self._prev_counts
        released = max(0, (prev_c.get(self.DIRECTIONS[action], 0) -
                           counts.get(self.DIRECTIONS[action], 0)))

        reward = self._env_helper.compute_reward_from_counts(
            action      = action,
            counts      = prev_c,
            wait_steps  = {d: self.wait_steps[d] for d in self.DIRECTIONS},
            released    = released,
        )

        # Monopoly penalty (mirroring TrafficEnv logic)
        if len(self._last_actions) == 3 and len(set(self._last_actions)) == 1:
            reward -= TrafficEnv.MONOPOLY_PENALTY

        # --- Q-table update ---
        self.agent.update(self._prev_state, self._prev_action, reward, nxt_state)

        # Update prev_counts for the NEXT feedback call (step-by-step) ← FIX #5
        self._prev_counts = counts.copy()
        self._prev_state  = nxt_state

        # Periodic checkpoint (FIX #12)
        if self.agent.step_count % self.save_every == 0:
            self.agent.save(self.save_path)

        # FIX #6 — decay epsilon during online learning
        if self.agent.step_count % 50 == 0:
            self.agent.decay_epsilon()

    # ──────────────────────────────────────────────────── starvation override
    def _check_starvation(self):
        """
        FIX #8 — only trigger override at 2× the limit (not 1×),
        so the RL agent has more chances to solve starvation itself.
        """
        worst_dir  = None
        worst_wait = 0
        for d in self.DIRECTIONS:
            w = self.wait_steps[d]
            if w > self.STARVATION_LIM * 2:
                if w > worst_wait:
                    worst_wait = w
                    worst_dir  = d
        if worst_dir:
            return self.DIRECTIONS.index(worst_dir)
        return None

    # ─────────────────────────────────────────────────────────── helpers
    def _count_waiting(self, vehicles):
        result = {}
        for d in self.DIRECTIONS:
            total = 0
            for lane in range(3):
                for v in vehicles[d][lane]:
                    if v.crossed == 0:
                        total += 1
            result[d] = total
        return result

    def _get_wait_bins(self):
        """Bin raw wait-steps the same way TrafficEnv does."""
        bins = {}
        for d in self.DIRECTIONS:
            w = self.wait_steps[d]
            if w <= 3:
                bins[d] = 0
            elif w <= 8:
                bins[d] = 1
            elif w <= self.STARVATION_LIM:
                bins[d] = 2
            else:
                bins[d] = 3
        return bins

    # ─────────────────────────────────────────────────────────── HUD metrics
    @property
    def metrics(self):
        state = self.agent.last_state
        qvals = self.agent.get_q_values(state)
        waits = [self.wait_steps[d] for d in self.DIRECTIONS]
        return {
            "state"        : state,
            "action"       : self.agent.last_action,
            "reward"       : self.agent.last_reward,
            "total_reward" : self.agent.total_reward,
            "steps"        : self.agent.step_count,
            "epsilon"      : self.agent.epsilon,
            "was_explore"  : self.agent.was_explore,
            "q_values"     : qvals,
            "wait_steps"   : waits,
        }