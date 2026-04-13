import argparse

from traffic_env import TrafficEnv
from rl_agent    import QLearningAgent


# ── training ──────────────────────────────────────────────────────────────────

def train(
    episodes=1000,
    steps_per_ep=300,
    save_path="qtable.json",
    save_every=200,
    arrival_probs=None,
):
    env   = TrafficEnv(
        arrival_probs=arrival_probs or [0.35, 0.35, 0.35, 0.35],
        max_steps=steps_per_ep,
    )
    agent = QLearningAgent(epsilon=1.0)
    agent.load(save_path)           # resume from checkpoint if available

    print(f"\n{'='*55}")
    print(f"  Q-Learning Traffic Controller — Training")
    print(f"  Episodes     : {episodes}")
    print(f"  Steps / ep   : {steps_per_ep}")
    print(f"  Save every   : {save_every} episodes → {save_path}")
    print(f"{'='*55}\n")

    best_reward = float("-inf")

    for ep in range(1, episodes + 1):

        # ── episode reset (FIX #11) ─────────────────────────────────────────
        counts, wait_bins = env.reset()
        state = agent.encode_state(counts, wait_bins)

        ep_reward = 0.0

        for _ in range(steps_per_ep):

            # 1. Select action
            action = agent.select_action(state)

            # 2. Step environment
            counts, wait_bins, reward, done, _ = env.step(action)

            # 3. Encode next state (identical encoding to RLBridge)
            next_state = agent.encode_state(counts, wait_bins)

            # 4. Update Q-table
            agent.update(state, action, reward, next_state)

            state      = next_state
            ep_reward += reward

            if done:
                break

        # ── epsilon decay (FIX #6) ─────────────────────────────────────────
        agent.decay_epsilon()

        if ep_reward > best_reward:
            best_reward = ep_reward

        # ── logging ────────────────────────────────────────────────────────
        if ep % 100 == 0 or ep == 1:
            print(
                f"  Ep {ep:5d}  |  reward={ep_reward:9.2f}  "
                f"best={best_reward:9.2f}  "
                f"ε={agent.epsilon:.4f}  "
                f"states={len(agent.q_table)}"
            )

        # ── periodic checkpoint (FIX #12) ──────────────────────────────────
        if ep % save_every == 0:
            agent.save(save_path)

    # final save
    agent.save(save_path)
    print(f"\n[train] Finished. Best episode reward: {best_reward:.2f}")
    return agent


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(episodes=20, steps_per_ep=300, load_path="qtable.json"):
    """Run the trained agent greedily and report average reward."""
    env   = TrafficEnv(max_steps=steps_per_ep)
    agent = QLearningAgent(epsilon=0.0)   # greedy
    agent.load(load_path)

    rewards = []
    for ep in range(1, episodes + 1):
        counts, wait_bins = env.reset()
        state  = agent.encode_state(counts, wait_bins)
        total  = 0.0
        for _ in range(steps_per_ep):
            action = agent.select_action(state)
            counts, wait_bins, reward, done, _ = env.step(action)
            state  = agent.encode_state(counts, wait_bins)
            total += reward
            if done:
                break
        rewards.append(total)
        if ep % 5 == 0:
            print(f"  Eval ep {ep:3d}  reward={total:.2f}")

    avg = sum(rewards) / len(rewards)
    print(f"\n[eval] Average reward over {episodes} episodes: {avg:.2f}")
    return avg


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-Learning traffic controller")
    parser.add_argument("--episodes",   type=int,  default=1000,
                        help="Number of training episodes")
    parser.add_argument("--steps",      type=int,  default=300,
                        help="Steps per episode")
    parser.add_argument("--save",       type=str,  default="qtable.json",
                        help="Checkpoint file path")
    parser.add_argument("--save-every", type=int,  default=200,
                        help="Save checkpoint every N episodes")
    parser.add_argument("--eval",       action="store_true",
                        help="Evaluate a saved model (skip training)")
    args = parser.parse_args()

    if args.eval:
        evaluate(load_path=args.save)
    else:
        train(
            episodes=args.episodes,
            steps_per_ep=args.steps,
            save_path=args.save,
            save_every=args.save_every,
        )