import numpy as np

# Environment Configuration
ENV_CONFIG = {
    'max_episodes': 10000, #10000
    'time_steps_per_episode': 24,
    'battery_capacity': 100,  # kWh
    'battery_power_rating': 50,  # kW
    'voltage_limits': [0.95, 1.05],
}

# Training Configuration
TRAINING_CONFIG = {
    'total_timesteps': 100000, #100000
    'learning_rate': 3e-4,
    'batch_size': 64,
    'buffer_size': 50000,
    'learning_starts': 1000,
    'target_update_interval': 1000,
    'exploration_fraction': 0.1,
    'exploration_final_eps': 0.02,
    'train_freq': 4,
    'gradient_steps': 1,
    'gamma': 0.99,
    'tau': 0.005,
}

# Hyperparameter ranges for optimization
HYPERPARAMETER_RANGES = {
    'learning_rate': [1e-5, 1e-3],
    'batch_size': [32, 128],
    'buffer_size': [10000, 100000],
    'gamma': [0.95, 0.999],
    'tau': [0.001, 0.01],
}

# Reward function weights
REWARD_WEIGHTS = {
    'voltage_regulation': 0.4,
    'power_loss': 0.3,
    'battery_utilization': 0.2,
    'solar_integration': 0.1,
}

# Network parameters
NETWORK_CONFIG = {
    'n_buses': 12,  # 2 HV/LV buses + 10 feeder nodes
    'pv_penetration': 0.4,  # 4 out of 10 feeder nodes have PV
    'load_scaling_factor': 1.0,
    'base_voltage': 0.4,  # kV (LV feeder voltage)
}
