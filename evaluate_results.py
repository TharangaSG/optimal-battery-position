import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO, DQN
import pandapower as pp
import os

from feeder_environment import DistributionFeederEnv
from config import ENV_CONFIG
from dqn_agent import DQNAgent

class ResultAnalyzer:
    """Analyze and visualize RL training results"""
    
    def __init__(self, model_path: str, algorithm: str = 'PPO'):
        self.algorithm = algorithm
        self.env = DistributionFeederEnv(ENV_CONFIG)
        
        # Load trained model
        if algorithm in ['PPO', 'DQN']:
            if algorithm == 'PPO':
                self.model = PPO.load(model_path)
            else:
                self.model = DQN.load(model_path)
        else:
            # Custom DQN
            state_dim = self.env.observation_space.shape[0]
            action_dim = 34
            self.model = DQNAgent(state_dim, action_dim, {})
            self.model.load(model_path)
    
    def analyze_battery_placement_strategy(self, num_episodes: int = 50):
        """Analyze the learned battery placement strategy"""
        placement_data = []
        performance_data = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_data = {
                'episode': episode,
                'positions': [],
                'soc_levels': [],
                'voltages': [],
                'rewards': [],
                'power_losses': []
            }
            
            for step in range(self.env.time_steps_per_episode):
                if hasattr(self.model, 'predict'):
                    action, _ = self.model.predict(state, deterministic=True)
                else:
                    action = self.model.act(state, training=False)
                
                state, reward, done, info = self.env.step(action)
                
                episode_data['positions'].append(info['battery_position'])
                episode_data['soc_levels'].append(info['battery_soc'])
                episode_data['rewards'].append(reward)
                episode_data['power_losses'].append(info['power_losses'])
                
                # Get voltage data
                try:
                    pp.runpp(self.env.net, verbose=False)
                    voltages = self.env.net.res_bus.vm_pu.values
                    episode_data['voltages'].append(voltages)
                except:
                    episode_data['voltages'].append(np.ones(self.env.n_buses))
                
                if done:
                    break
            
            placement_data.append(episode_data)
        
        return self._analyze_placement_patterns(placement_data)
    
    def _analyze_placement_patterns(self, placement_data):
        """Analyze battery placement patterns"""
        # Extract placement frequencies
        all_positions = []
        for episode_data in placement_data:
            all_positions.extend(episode_data['positions'])
        
        position_counts = pd.Series(all_positions).value_counts().sort_index()
        
        # Analyze temporal patterns
        hourly_positions = np.zeros((24, self.env.n_buses))
        for episode_data in placement_data:
            for hour, position in enumerate(episode_data['positions']):
                if hour < 24:
                    hourly_positions[hour, position] += 1
        
        # Normalize by number of episodes
        hourly_positions = hourly_positions / len(placement_data)
        
        return {
            'position_frequencies': position_counts,
            'hourly_patterns': hourly_positions,
            'placement_data': placement_data
        }
    
    def visualize_results(self, analysis_results):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Battery Position Frequency
        plt.subplot(2, 3, 1)
        analysis_results['position_frequencies'].plot(kind='bar')
        plt.title('Battery Placement Frequency by Bus')
        plt.xlabel('Bus Number')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        
        # 2. Hourly Placement Heatmap
        plt.subplot(2, 3, 2)
        sns.heatmap(analysis_results['hourly_patterns'], 
                   cmap='YlOrRd', cbar_kws={'label': 'Placement Probability'})
        plt.title('Hourly Battery Placement Patterns')
        plt.xlabel('Bus Number')
        plt.ylabel('Hour of Day')
        
        # 3. SOC Patterns
        plt.subplot(2, 3, 3)
        soc_data = []
        for episode_data in analysis_results['placement_data'][:10]:
            soc_data.append(episode_data['soc_levels'])
        
        for i, soc in enumerate(soc_data):
            plt.plot(soc, alpha=0.7, label=f'Episode {i+1}')
        plt.title('Battery SOC Patterns (First 10 Episodes)')
        plt.xlabel('Time Step')
        plt.ylabel('SOC')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Reward Evolution
        plt.subplot(2, 3, 4)
        reward_data = []
        for episode_data in analysis_results['placement_data']:
            reward_data.append(np.sum(episode_data['rewards']))
        
        plt.plot(reward_data)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        # 5. Power Loss Analysis
        plt.subplot(2, 3, 5)
        loss_data = []
        for episode_data in analysis_results['placement_data']:
            loss_data.append(np.mean(episode_data['power_losses']))
        
        plt.plot(loss_data)
        plt.title('Average Power Losses per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Power Loss (MW)')
        
        # 6. Voltage Profile Analysis
        plt.subplot(2, 3, 6)
        voltage_violations = []
        for episode_data in analysis_results['placement_data']:
            violations = 0
            for voltages in episode_data['voltages']:
                violations += np.sum((voltages < 0.95) | (voltages > 1.05))
            voltage_violations.append(violations)
        
        plt.plot(voltage_violations)
        plt.title('Voltage Violations per Episode')
        plt.xlabel('Episode')
        plt.ylabel('Number of Violations')
        
        plt.tight_layout()
        plt.savefig('battery_placement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_with_baseline(self):
        """Compare RL strategy with baseline methods"""
        # Baseline 1: Fixed position (middle of feeder)
        baseline_fixed = self._evaluate_fixed_position(16)
        
        # Baseline 2: Random placement
        baseline_random = self._evaluate_random_placement()
        
        # RL strategy
        rl_results = self._evaluate_rl_strategy()
        
        # Create comparison
        comparison_df = pd.DataFrame({
            'Strategy': ['Fixed Position', 'Random Placement', 'RL Strategy'],
            'Avg Reward': [baseline_fixed['avg_reward'], 
                          baseline_random['avg_reward'], 
                          rl_results['avg_reward']],
            'Avg Violations': [baseline_fixed['avg_violations'], 
                              baseline_random['avg_violations'], 
                              rl_results['avg_violations']],
            'Avg Losses': [baseline_fixed['avg_losses'], 
                          baseline_random['avg_losses'], 
                          rl_results['avg_losses']]
        })
        
        return comparison_df
    
    def _evaluate_fixed_position(self, position: int, num_episodes: int = 20):
        """Evaluate fixed battery position strategy"""
        rewards = []
        violations = []
        losses = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_violations = 0
            episode_losses = 0
            
            for step in range(self.env.time_steps_per_episode):
                # Fixed action: place battery at fixed position with optimal charging
                action = np.array([position, 0])  # No charging for simplicity
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_violations += info['voltage_violations']
                episode_losses += info['power_losses']
                
                if done:
                    break
            
            rewards.append(episode_reward)
            violations.append(episode_violations)
            losses.append(episode_losses)
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_violations': np.mean(violations),
            'avg_losses': np.mean(losses)
        }
    
    def _evaluate_random_placement(self, num_episodes: int = 20):
        """Evaluate random battery placement strategy"""
        rewards = []
        violations = []
        losses = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_violations = 0
            episode_losses = 0
            
            for step in range(self.env.time_steps_per_episode):
                # Random action
                position = np.random.randint(0, self.env.n_buses)
                power = np.random.uniform(-50, 50)
                action = np.array([position, power])
                
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_violations += info['voltage_violations']
                episode_losses += info['power_losses']
                
                if done:
                    break
            
            rewards.append(episode_reward)
            violations.append(episode_violations)
            losses.append(episode_losses)
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_violations': np.mean(violations),
            'avg_losses': np.mean(losses)
        }
    
    def _evaluate_rl_strategy(self, num_episodes: int = 20):
        """Evaluate RL strategy"""
        rewards = []
        violations = []
        losses = []
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_violations = 0
            episode_losses = 0
            
            for step in range(self.env.time_steps_per_episode):
                if hasattr(self.model, 'predict'):
                    action, _ = self.model.predict(state, deterministic=True)
                else:
                    action = self.model.act(state, training=False)
                
                state, reward, done, info = self.env.step(action)
                
                episode_reward += reward
                episode_violations += info['voltage_violations']
                episode_losses += info['power_losses']
                
                if done:
                    break
            
            rewards.append(episode_reward)
            violations.append(episode_violations)
            losses.append(episode_losses)
        
        return {
            'avg_reward': np.mean(rewards),
            'avg_violations': np.mean(violations),
            'avg_losses': np.mean(losses)
        }

def main():
    """Main evaluation function"""
    # Specify model path and algorithm
    model_path = "models/PPO_battery_placement_20241201_120000.zip"  # Update with actual path
    algorithm = "PPO"
    
    analyzer = ResultAnalyzer(model_path, algorithm)
    
    # Analyze battery placement strategy
    print("Analyzing battery placement strategy...")
    analysis_results = analyzer.analyze_battery_placement_strategy()
    
    # Visualize results
    print("Creating visualizations...")
    analyzer.visualize_results(analysis_results)
    
    # Compare with baselines
    print("Comparing with baseline methods...")
    comparison_df = analyzer.compare_with_baseline()
    print(comparison_df)
    
    # Save results
    comparison_df.to_csv('strategy_comparison.csv', index=False)
    print("Analysis completed! Results saved.")

if __name__ == "__main__":
    main()
