import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.stack, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def update(self, batch_size: int = 32):
        if len(self.replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        q_values = self.policy_net(states).gather(1, actions)
        next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        expected_q = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

def contrastive_loss(emb_bug: torch.Tensor, emb_stack: torch.Tensor, emb_random: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    cos_sim_pos = F.cosine_similarity(emb_bug, emb_stack) / temperature
    cos_sim_neg = F.cosine_similarity(emb_bug, emb_random) / temperature
    loss = -torch.log(torch.exp(cos_sim_pos) / (torch.exp(cos_sim_pos) + torch.exp(cos_sim_neg) + 1e-8))
    return loss.mean()

class AtlasAgent:
    def __init__(self, file_state_dim: int, function_state_dim: int, line_state_dim: int,
                 num_files: int, num_functions: int, num_lines: int):
        self.file_agent = DQNAgent(state_dim=file_state_dim, action_dim=num_files)
        self.func_agent = DQNAgent(state_dim=function_state_dim, action_dim=num_functions)
        self.line_agent = DQNAgent(state_dim=line_state_dim, action_dim=num_lines)

    def train(self, env, episodes: int = 1000, update_target_every: int = 10, batch_size: int = 32):
        for ep in range(episodes):
            file_states = env.reset_file_level()
            file_action = self.file_agent.select_action(file_states)
            file_reward, file_done, _ = env.step_file_level(file_action)
            func_states = env.get_function_states()
            func_action = self.func_agent.select_action(func_states)
            func_reward, func_done, _ = env.step_function_level(func_action)
            line_states = env.get_line_states()
            line_action = self.line_agent.select_action(line_states)
            line_reward, line_done, _ = env.step_line_level(line_action)
            self.file_agent.replay_buffer.push(file_states[file_action], file_action, file_reward, file_states[file_action], file_done)
            self.func_agent.replay_buffer.push(func_states[func_action], func_action, func_reward, func_states[func_action], func_done)
            self.line_agent.replay_buffer.push(line_states[line_action], line_action, line_reward, line_states[line_action], line_done)
            self.file_agent.update(batch_size)
            self.func_agent.update(batch_size)
            self.line_agent.update(batch_size)
            if ep % update_target_every == 0:
                self.file_agent.update_target()
                self.func_agent.update_target()
                self.line_agent.update_target()
            emb_bug = torch.FloatTensor(np.random.rand(1, 768))
            emb_stack = torch.FloatTensor(np.random.rand(1, 768))
            emb_random = torch.FloatTensor(np.random.rand(1, 768))
            cl_loss = contrastive_loss(emb_bug, emb_stack, emb_random)
            if ep % 50 == 0:
                print(f"Episode {ep}: File Reward {file_reward:.2f}, Function Reward {func_reward:.2f}, Line Reward {line_reward:.2f}, Contrastive Loss {cl_loss.item():.4f}")

    def evaluate(self, env, num_episodes: int = 100) -> dict:
        total_file_reward = 0
        total_func_reward = 0
        total_line_reward = 0
        success_count = 0
        for _ in range(num_episodes):
            file_states = env.reset_file_level()
            file_action = self.file_agent.select_action(file_states)
            file_reward, file_done, _ = env.step_file_level(file_action)
            func_states = env.get_function_states()
            func_action = self.func_agent.select_action(func_states)
            func_reward, func_done, _ = env.step_function_level(func_action)
            line_states = env.get_line_states()
            line_action = self.line_agent.select_action(line_states)
            line_reward, line_done, _ = env.step_line_level(line_action)
            total_file_reward += file_reward
            total_func_reward += func_reward
            total_line_reward += line_reward
            if file_done and func_done and line_done:
                success_count += 1
        return {
            "avg_file_reward": total_file_reward / num_episodes,
            "avg_function_reward": total_func_reward / num_episodes,
            "avg_line_reward": total_line_reward / num_episodes,
            "success_rate": success_count / num_episodes
        }
