import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List

class DQNNetwork(nn.Module):
    """Deep Q-Network for battery placement optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """DQN Agent for battery placement optimization"""
    
    def __init__(self, state_dim: int, action_dim: int, config: dict):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_dim, action_dim).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(), 
            lr=config['learning_rate']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config['buffer_size'])
        
        # Training parameters
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Update target network with current network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def soft_update_target_network(self):
        """Soft update target network"""
        for target_param, local_param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action for exploration - battery can only be on feeder buses (2-11)
            battery_position = random.randint(2, 11)  # 12 buses total, feeder buses are 2-11
            charge_power = random.uniform(-50, 50)  # Battery power rating
            return np.array([battery_position, charge_power])
        
        # Greedy action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        
        # Convert Q-values to actions (this is a simplified approach)
        # In practice, you might want to use a more sophisticated action selection
        # Map Q-values to valid feeder bus positions (2-11)
        feeder_q_values = q_values[0, :10]  # First 10 outputs for feeder buses
        battery_position = torch.argmax(feeder_q_values).item() + 2  # Add 2 to get bus index 2-11
        charge_power = torch.tanh(q_values[0, 10]) * 50  # Scale to [-50, 50]
        
        return np.array([battery_position, charge_power.item()])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self):
        """Train the DQN agent"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states)
        
        # Next Q-values from target network
        next_q_values = self.target_network(next_states).detach()
        target_q_values = rewards + (self.gamma * next_q_values.max(1)[0] * ~dones)
        
        # Compute loss (simplified - in practice you'd need to handle continuous actions)
        loss = nn.MSELoss()(current_q_values.max(1)[0], target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Soft update target network
        self.soft_update_target_network()
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
