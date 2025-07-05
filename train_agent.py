import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import torch
import os
from datetime import datetime

from feeder_environment import DistributionFeederEnv
from config import ENV_CONFIG, TRAINING_CONFIG
from dqn_agent import DQNAgent

class TrainingManager:
    """Manages the training process for the RL agent"""
    
    def __init__(self, algorithm='PPO'):
        self.algorithm = algorithm
        self.env_config = ENV_CONFIG
        self.training_config = TRAINING_CONFIG
        
        # Create directories
        self.model_dir = "models"
        self.log_dir = "logs"
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize environment
        self.env = DistributionFeederEnv(self.env_config)
        self.env = Monitor(self.env, self.log_dir)
        
    def train_with_stable_baselines(self):
        """Train using Stable Baselines3"""
        print(f"Training with {self.algorithm}")
        
        if self.algorithm == 'PPO':
            model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.training_config['learning_rate'],
                batch_size=self.training_config['batch_size'],
                gamma=self.training_config['gamma'],
                verbose=1,
                tensorboard_log=self.log_dir
            )
        elif self.algorithm == 'DQN':
            # Note: DQN in SB3 requires discrete action space
            # For continuous actions, we'd need to modify the environment
            model = DQN(
                "MlpPolicy",
                self.env,
                learning_rate=self.training_config['learning_rate'],
                buffer_size=self.training_config['buffer_size'],
                learning_starts=self.training_config['learning_starts'],
                batch_size=self.training_config['batch_size'],
                gamma=self.training_config['gamma'],
                verbose=1,
                tensorboard_log=self.log_dir
            )
        
        # Callbacks
        eval_callback = EvalCallback(
            self.env,
            best_model_save_path=self.model_dir,
            log_path=self.log_dir,
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        model.learn(
            total_timesteps=self.training_config['total_timesteps'],
            callback=eval_callback
        )
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model.save(f"{self.model_dir}/{self.algorithm}_battery_placement_{timestamp}")
        
        return model
    
    def train_custom_dqn(self):
        """Train using custom DQN implementation"""
        print("Training with Custom DQN")
        
        # Initialize agent
        state_dim = self.env.observation_space.shape[0]
        action_dim = 11  # 10 feeder buses + 1 power action (simplified)
        
        agent = DQNAgent(state_dim, action_dim, self.training_config)
        
        # Training loop
        episode_rewards = []
        episode_losses = []
        
        for episode in range(1000):
            state = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            
            for step in range(self.env.time_steps_per_episode):
                # Select action
                action = agent.act(state, training=True)
                
                # Take step
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, done)
                
                # Train agent
                if len(agent.replay_buffer) > agent.batch_size:
                    loss = agent.train()
                    if loss is not None:
                        episode_loss += loss
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)
            
            # Logging
            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                avg_loss = np.mean(episode_losses[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, "
                      f"Avg Loss: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
            
            # Save model periodically
            if episode % 200 == 0:
                agent.save(f"{self.model_dir}/custom_dqn_episode_{episode}.pth")
        
        # Save final model
        agent.save(f"{self.model_dir}/custom_dqn_final.pth")
        
        return agent, episode_rewards, episode_losses
    
    def evaluate_model(self, model, num_episodes=10):
        """Evaluate trained model"""
        print("Evaluating model...")
        
        episode_rewards = []
        voltage_violations = []
        power_losses = []
        battery_positions = []
        
        # Access the underlying environment's attribute
        time_steps = self.env.env.time_steps_per_episode if hasattr(self.env, 'env') else 24 # type: ignore
        
        for episode in range(num_episodes):
            state = self.env.reset()
            # Handle both old gym and new gymnasium reset formats
            if isinstance(state, tuple):
                state = state[0]
                
            episode_reward = 0
            episode_violations = 0
            episode_losses = 0
            positions = []
            
            for step in range(time_steps):  # Use the extracted time_steps
                if hasattr(model, 'predict'):
                    # Stable Baselines3 model
                    action, _ = model.predict(state, deterministic=True)
                else:
                    # Custom DQN agent
                    action = model.act(state, training=False)
                
                step_result = self.env.step(action)
                
                # Handle both old gym (4 values) and new gymnasium (5 values) formats
                if len(step_result) == 5:
                    state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    state, reward, done, info = step_result # type: ignore
                
                episode_reward += reward # type: ignore
                episode_violations += info['voltage_violations']
                episode_losses += info['power_losses']
                positions.append(info['battery_position'])
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            voltage_violations.append(episode_violations)
            power_losses.append(episode_losses)
            battery_positions.append(positions)
        
        # Print results
        print(f"Average Episode Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Voltage Violations: {np.mean(voltage_violations):.2f}")
        print(f"Average Power Losses: {np.mean(power_losses):.4f} MW")
        
        from collections import Counter

        # Flatten the list of positions from all episodes
        all_positions = []
        for episode_positions in battery_positions:
            all_positions.extend(episode_positions)

        # Count frequency of each bus
        if all_positions:
            position_counts = Counter(all_positions)
            optimal_node, count = position_counts.most_common(1)[0]
            print(f"\nOptimal node for battery placement: Bus {optimal_node} (chosen {count} times)")
            print("Node selection frequency:", position_counts.most_common())
        else:
            print("No battery positions recorded during evaluation.")

        return {
            'rewards': episode_rewards,
            'violations': voltage_violations,
            'losses': power_losses,
            'positions': battery_positions
        }

    
    def plot_training_results(self, rewards, losses=None):
        """Plot training results"""
        fig, axes = plt.subplots(1, 2 if losses is not None else 1, figsize=(15, 5))
        
        if losses is not None:
            axes[0].plot(rewards)
            axes[0].set_title('Episode Rewards')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
            axes[0].grid(True)
            
            axes[1].plot(losses)
            axes[1].set_title('Training Loss')
            axes[1].set_xlabel('Episode')
            axes[1].set_ylabel('Loss')
            axes[1].grid(True)
        else:
            axes.plot(rewards)
            axes.set_title('Episode Rewards')
            axes.set_xlabel('Episode')
            axes.set_ylabel('Reward')
            axes.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.log_dir}/training_results.png")
        plt.show()

def main():
    """Main training function"""
    # Choose algorithm: 'PPO', 'DQN', or 'Custom_DQN'
    algorithm = 'PPO'
    
    trainer = TrainingManager(algorithm)
    
    if algorithm in ['PPO', 'DQN']:
        model = trainer.train_with_stable_baselines()
        results = trainer.evaluate_model(model)
    elif algorithm == 'Custom_DQN':
        agent, rewards, losses = trainer.train_custom_dqn()
        trainer.plot_training_results(rewards, losses)
        results = trainer.evaluate_model(agent)
    
    print("Training completed!")
    return model if algorithm in ['PPO', 'DQN'] else agent

if __name__ == "__main__":
    main()
