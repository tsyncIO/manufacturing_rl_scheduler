# manufacturing_rl/train.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from manufacturing_rl.agents import QLearningAgent
from manufacturing_rl.simulator import ManufacturingSimulator

def train_agent(agent: QLearningAgent, simulator: ManufacturingSimulator, num_episodes: int, batch_size: int) -> Tuple[List[float], List[float]]:
    all_rewards: List[float] = []
    avg_rewards: List[float] = []

    for episode in range(num_episodes):
        state = simulator.reset()
        done = False
        total_reward = 0
        step = 0

        while not done and step < simulator.max_steps:
            action = agent.select_action(state)
            next_state, reward, done = simulator.step(action)
            agent.store_transition(state, action, reward, next_state)
            state = next_state
            total_reward += reward
            step += 1
            if len(agent.memory) > batch_size:
                agent.update_model(batch_size)

        all_rewards.append(total_reward)
        avg_rewards.append(np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards))
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Avg Reward: {avg_rewards[-1]:.2f}, Epsilon: {agent.epsilon:.4f}")

    return all_rewards, avg_rewards

def visualize_manufacturing_schedule(agent: QLearningAgent, simulator: ManufacturingSimulator, num_steps: int = 20) -> None:
    state = simulator.reset()
    done = False
    step = 0

    print("\n--- Manufacturing Schedule ---")
    job_names = ["Engine Block", "Transmission", "Body Panel", "Interior", "Final Inspect"]
    print(f"Initial Job Quantities: {dict(zip(job_names, simulator.job_quantities))}")
    print(f"Job Deadlines: {dict(zip(job_names, simulator.job_deadlines))}")
    print(f"Job Priorities: {dict(zip(job_names, simulator.job_priorities))}")

    while not done and step < num_steps:
        action = agent.select_action(state, explore=False)
        next_state, reward, done = simulator.step(action)

        print(f"\nTime Slot {step + 1}:")
        assigned_job = job_names[action]
        assigned_machine = simulator.job_machine_assignments[action]
        print(f"  Action: Schedule '{assigned_job}' on Machine {assigned_machine + 1}")
        print(f"  Remaining Quantities: {dict(zip(job_names, simulator.job_quantities))}")
        print(f"  Machine Availability: {simulator.machine_available_time}")
        print(f"  Reward: {reward:.2f}")

        state = next_state
        step += 1

    total_tardiness = np.sum(np.maximum(0, simulator