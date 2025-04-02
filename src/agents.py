# manufacturing_rl/agents.py
import torch
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import List, Tuple
from manufacturing_rl.models import QNetwork

class QLearningAgent:
    def __init__(self, state_dim: int, num_actions: int, learning_rate: float = 0.001, discount_factor: float = 0.9,
                 target_update_freq: int = 10,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01, epsilon_decay: float = 0.00005,
                 memory_capacity: int = 1000) -> None:
        self.policy_net = QNetwork(state_dim, num_actions)
        self.target_net = QNetwork(state_dim, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.target_update_freq = target_update_freq
        self.num_actions = num_actions
        self.memory: deque[Tuple[np.ndarray, int, float, np.ndarray]] = deque(maxlen=memory_capacity)
        self.train_steps = 0
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> None:
        self.memory.append((state, action, reward, next_state))

    def update_model(self, batch_size: int) -> None:
        if len(self.memory) < batch_size:
            return

        actual_batch_size = min(batch_size, len(self.memory))
        batch = random.sample(self.memory, actual_batch_size)

        states, actions, rewards, next_states = zip(*batch)

        states_np = np.array(states)
        next_states_np = np.array(next_states)
        
        state_tensor = torch.FloatTensor(states_np)
        action_tensor = torch.LongTensor(np.array(actions)).unsqueeze(1)
        reward_tensor = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_state_tensor = torch.FloatTensor(next_states_np)

        q_values = self.policy_net(state_tensor).gather(1, action_tensor)
        next_q_values = self.target_net(next_state_tensor).max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + self.discount_factor * next_q_values

        loss = nn.functional.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)