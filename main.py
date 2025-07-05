#!/usr/bin/env python3
"""
Main execution script for Battery Placement Optimization using Reinforcement Learning
"""

import argparse
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_agent import TrainingManager
from evaluate_results import ResultAnalyzer
from config import ENV_CONFIG, TRAINING_CONFIG

def main():
    parser = argparse.ArgumentParser(description='Battery Placement Optimization using RL')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'both'], 
                       default='both', help='Mode to run')
    parser.add_argument('--algorithm', choices=['PPO', 'DQN', 'Custom_DQN'], 
                       default='PPO', help='RL algorithm to use')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='Path to trained model for evaluation')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=50, 
                       help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Battery Placement Optimization using Reinforcement Learning")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Training Episodes: {args.episodes}")
    print("=" * 60)
    
    # Update config with command line arguments
    TRAINING_CONFIG['total_timesteps'] = args.episodes * ENV_CONFIG['time_steps_per_episode']
    
    if args.mode in ['train', 'both']:
        print("\n🚀 Starting Training Phase...")
        trainer = TrainingManager(args.algorithm)
        
        if args.algorithm in ['PPO', 'DQN']:
            model = trainer.train_with_stable_baselines()
            model_path = f"models/{args.algorithm}_battery_placement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model.save(model_path)
        else:
            agent, rewards, losses = trainer.train_custom_dqn()
            model_path = f"models/custom_dqn_final.pth"
            trainer.plot_training_results(rewards, losses)
        
        print(f"✅ Training completed! Model saved to: {model_path}")
        
        # Quick evaluation
        if args.algorithm in ['PPO', 'DQN']:
            results = trainer.evaluate_model(model, num_episodes=10)
        else:
            results = trainer.evaluate_model(agent, num_episodes=10)
        
        print("\n📊 Quick Training Results:")
        print(f"Average Reward: {sum(results['rewards'])/len(results['rewards']):.2f}")
        print(f"Average Violations: {sum(results['violations'])/len(results['violations']):.2f}")
    
    if args.mode in ['evaluate', 'both']:
        print("\n🔍 Starting Evaluation Phase...")
        
        if args.mode == 'evaluate' and args.model_path is None:
            print("❌ Error: Model path required for evaluation mode")
            return
        
        if args.mode == 'both':
            eval_model_path = model_path
        else:
            eval_model_path = args.model_path
        
        # Comprehensive analysis
        analyzer = ResultAnalyzer(eval_model_path, args.algorithm)
        
        print("Analyzing battery placement strategy...")
        analysis_results = analyzer.analyze_battery_placement_strategy(args.eval_episodes)
        
        print("Creating visualizations...")
        analyzer.visualize_results(analysis_results)
        
        print("Comparing with baseline methods...")
        comparison_df = analyzer.compare_with_baseline()
        print("\n📈 Strategy Comparison Results:")
        print(comparison_df.to_string(index=False))
        
        # Save detailed results
        comparison_df.to_csv('strategy_comparison.csv', index=False)
        
        print("\n✅ Evaluation completed!")
        print("📁 Results saved:")
        print("   - battery_placement_analysis.png")
        print("   - strategy_comparison.csv")
        print("   - Training logs in logs/ directory")
    
    print("\n🎉 All tasks completed successfully!")

if __name__ == "__main__":
    main()
