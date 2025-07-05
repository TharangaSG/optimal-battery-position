import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

class DistributionFeederEnv(gym.Env):
    """
    Custom Environment for Battery Placement Optimization in Distribution Feeders
    """
    
    def __init__(self, config: Dict):
        super(DistributionFeederEnv, self).__init__()

        # Environment configuration
        self.config = config
        self.max_episodes = config.get('max_episodes', 1000)
        self.time_steps_per_episode = config.get('time_steps_per_episode', 24)
        self.battery_capacity = config.get('battery_capacity', 100)  # kWh
        self.battery_power_rating = config.get('battery_power_rating', 50)  # kW
        self.voltage_limits = config.get('voltage_limits', [0.95, 1.05])
        
        # Create distribution network
        self.net = self._create_distribution_network()
        self.n_buses = len(self.net.bus)
        
        # State space: [bus_voltages, bus_loads, pv_generation, battery_soc, time_of_day]
        self.state_dim = self.n_buses * 3 + 2  # voltages, loads, pv + soc + time
        
        # Action space: [battery_bus_position, charge_discharge_power]
        # Battery can only be placed on feeder buses (buses 2-11, which are the 10 feeder nodes)
        self.action_space = spaces.Box(
            low=np.array([2, -self.battery_power_rating]),  # Start from bus 2 (first feeder node)
            high=np.array([self.n_buses-1, self.battery_power_rating]),
            dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.state_dim,), dtype=np.float32
        )
        
        # Initialize state variables
        self.current_step = 0
        self.battery_soc = 0.5  # Start at 50% SOC
        self.battery_position = 0
        self.load_profiles = self._generate_load_profiles()
        self.pv_profiles = self._generate_pv_profiles()
        
    def _create_distribution_network(self) -> pp.pandapowerNet:
        """Create a custom distribution network based on feeder_1.py"""
        # Create empty network
        net = pp.create_empty_network()

        # High voltage bus (11kV)
        hv_bus = pp.create_bus(net, vn_kv=11.0, name="HV Bus 11kV")

        # Low voltage bus at transformer secondary (400V)
        lv_main_bus = pp.create_bus(net, vn_kv=0.4, name="LV Main Bus 400V")

        # Create 10 feeder nodes (400V each)
        feeder_buses = []
        for i in range(10):
            bus = pp.create_bus(net, vn_kv=0.4, name=f"Node_{i+1}")
            feeder_buses.append(bus)

        # Create external grid connection at 11kV
        pp.create_ext_grid(net, bus=hv_bus, vm_pu=1.0, name="Grid Connection")

        # Create 11kV to 400V transformer
        pp.create_transformer(net, 
                             hv_bus=hv_bus, 
                             lv_bus=lv_main_bus, 
                             std_type="0.25 MVA 10/0.4 kV",
                             name="11kV/400V Transformer")

        # Create feeder lines connecting main bus to each node
        line_lengths = [0.036, 0.02, 0.02, 0.025, 0.025, 0.026, 0.028, 0.023, 0.0225, 0.0225]  # km

        for i, (bus, length) in enumerate(zip(feeder_buses, line_lengths)):
            if i == 0:
                # First node connects to main bus
                pp.create_line(net, 
                              from_bus=lv_main_bus, 
                              to_bus=bus, 
                              length_km=length,
                              std_type="NAYY 4x50 SE",
                              name=f"Feeder_Line_{i+1}")
            else:
                # Subsequent nodes connect in series (radial feeder)
                pp.create_line(net, 
                              from_bus=feeder_buses[i-1], 
                              to_bus=bus, 
                              length_km=length,
                              std_type="NAYY 4x50 SE",
                              name=f"Feeder_Line_{i+1}")

        # Create house loads for Sri Lankan residential consumers
        house_loads = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  # kW
        load_power_factors = [0.9] * 10

        for i, (bus, p_load, pf) in enumerate(zip(feeder_buses, house_loads, load_power_factors)):
            q_load = p_load * np.tan(np.arccos(pf))  # Calculate reactive power
            pp.create_load(net, 
                           bus=bus, 
                           p_mw=p_load/1000,  # Convert kW to MW
                           q_mvar=q_load/1000,  # Convert kVAR to MVAR
                           name=f"House_Load_{i+1}")

        # Create rooftop solar systems 
        solar_nodes = [4, 5, 6, 7]  # Indices for poles 5, 6, 7, 8  
        solar_capacities = [17, 5.0, 16, 3.3]  # kW

        for i, (node_idx, solar_kw) in enumerate(zip(solar_nodes, solar_capacities)):
            pp.create_sgen(net, 
                           bus=feeder_buses[node_idx], 
                           p_mw=solar_kw/1000,  # Convert kW to MW
                           q_mvar=0,  # Unity power factor for solar
                           name=f"Rooftop_Solar_{node_idx+1}",
                           type="PV")

        # Add additional solar system to Node 7 (pole 8)
        pp.create_sgen(net, 
                       bus=feeder_buses[6],  # Node 7 (index 6)
                       p_mw=2.5/1000,  # 2.5 kW additional solar system
                       q_mvar=0,  # Unity power factor
                       name="Solar_System_Node7_Additional",
                       type="PV")
        
        return net
    
    def _generate_load_profiles(self) -> np.ndarray:
        """Generate realistic load profiles for each bus"""
        np.random.seed(42)
        n_time_steps = self.time_steps_per_episode
        
        # Base load pattern (typical daily curve)
        base_pattern = np.array([
            0.6, 0.5, 0.4, 0.4, 0.4, 0.5, 0.7, 0.9,
            1.0, 0.9, 0.8, 0.8, 0.9, 1.0, 1.0, 1.1,
            1.2, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7
        ])
        
        # Add randomness and scale for each bus
        load_profiles = np.zeros((self.n_buses, n_time_steps))
        for bus in range(self.n_buses):
            scale_factor = 0.5 + np.random.random() * 1.0
            noise = np.random.normal(0, 0.1, n_time_steps)
            load_profiles[bus] = (base_pattern + noise) * scale_factor
            load_profiles[bus] = np.maximum(load_profiles[bus], 0.1)
        
        return load_profiles
    
    def _generate_pv_profiles(self) -> np.ndarray:
        """Generate PV generation profiles"""
        np.random.seed(43)
        n_time_steps = self.time_steps_per_episode
        
        # Solar irradiance pattern
        solar_pattern = np.array([
            0, 0, 0, 0, 0, 0, 0.1, 0.3,
            0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.6,
            0.4, 0.2, 0.1, 0, 0, 0, 0, 0
        ])
        
        # PV generation for each PV bus (matching the feeder_1.py network)
        # Buses 6, 7, 8, 9 (corresponding to feeder nodes 5, 6, 7, 8) + bus 8 has additional solar
        pv_buses = [6, 7, 8, 9]  # These correspond to solar_nodes [4, 5, 6, 7] in feeder_buses
        pv_profiles = np.zeros((self.n_buses, n_time_steps))
        
        # Main solar systems
        solar_capacities = [17, 5.0, 16, 3.3]  # kW capacities from feeder_1.py
        for i, (bus, capacity) in enumerate(zip(pv_buses, solar_capacities)):
            cloud_factor = 0.7 + np.random.random() * 0.3
            noise = np.random.normal(0, 0.05, n_time_steps)
            # Scale by capacity relative to base 0.1 MW
            capacity_factor = (capacity / 1000) / 0.1  # Convert kW to MW and normalize
            pv_profiles[bus] = np.maximum(
                (solar_pattern + noise) * cloud_factor * capacity_factor, 0
            )
        
        # Additional solar system at bus 8 (Node 7)
        additional_capacity = 2.5  # kW
        additional_factor = (additional_capacity / 1000) / 0.1
        cloud_factor = 0.7 + np.random.random() * 0.3
        noise = np.random.normal(0, 0.05, n_time_steps)
        pv_profiles[8] += np.maximum(
            (solar_pattern + noise) * cloud_factor * additional_factor, 0
        )
        
        return pv_profiles
    
    def reset(self, seed=None, options=None):
        """Reset the environment - gymnasium compatible"""
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        
        # Reset environment state
        self.current_step = 0
        self.battery_soc = 0.5
        self.battery_position = 0
        
        # Reset network to initial state
        try:
            self._update_network_state()
        except Exception as e:
            print(f"Warning: Network state update failed: {e}")
        
        # Get initial observation
        obs = self._get_observation()
        
        # Return observation and info dict (gymnasium format)
        info = {
            'battery_soc': self.battery_soc,
            'battery_position': self.battery_position,
            'current_step': self.current_step
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        # Parse action - battery can only be placed on feeder buses (2-11)
        battery_position = int(np.clip(action[0], 2, self.n_buses-1))
        charge_power = np.clip(action[1], -self.battery_power_rating, 
                            self.battery_power_rating)
        
        # Update battery position and SOC
        self.battery_position = battery_position
        self._update_battery_soc(charge_power)
        
        # Update network state
        self._update_network_state()
        
        # Run power flow
        try:
            pp.runpp(self.net, verbose=False)
            converged = True
        except:
            converged = False
        
        # Calculate reward
        reward = self._calculate_reward(converged)
        
        # Check if episode is done
        self.current_step += 1
        terminated = self.current_step >= self.time_steps_per_episode
        truncated = False  # Gymnasium requires this flag, set False if no truncation logic
        
        # Get next observation
        obs = self._get_observation()
        
        info = {
            'converged': converged,
            'voltage_violations': self._count_voltage_violations(),
            'power_losses': self._calculate_power_losses(),
            'battery_soc': self.battery_soc,
            'battery_position': self.battery_position
        }
        
        return obs, reward, terminated, truncated, info

    
    def _update_battery_soc(self, charge_power: float):
        """Update battery state of charge"""
        # Convert power to energy (assuming 1-hour time step)
        energy_change = charge_power / self.battery_capacity
        
        # Update SOC with efficiency considerations
        if charge_power > 0:  # Charging
            self.battery_soc += energy_change * 0.95  # 95% charging efficiency
        else:  # Discharging
            self.battery_soc += energy_change / 0.95  # 95% discharging efficiency
        
        # Clip SOC to valid range
        self.battery_soc = np.clip(self.battery_soc, 0.0, 1.0)
    
    def _update_network_state(self):
        """Update network loads and generation"""
        time_idx = self.current_step % self.time_steps_per_episode
        
        # Update loads
        for i, load_idx in enumerate(self.net.load.index):
            bus = self.net.load.loc[load_idx, 'bus']
            self.net.load.loc[load_idx, 'p_mw'] = self.load_profiles[bus, time_idx] * 0.1
        
        # Update PV generation
        for i, sgen_idx in enumerate(self.net.sgen.index):
            bus = self.net.sgen.loc[sgen_idx, 'bus']
            self.net.sgen.loc[sgen_idx, 'p_mw'] = self.pv_profiles[bus, time_idx] * 0.1
        
        # Update battery (remove existing battery and add new one)
        battery_indices = self.net.storage.index[self.net.storage.name.str.contains('Battery', na=False)]
        if len(battery_indices) > 0:
            self.net.storage.drop(battery_indices, inplace=True)
        
        # Add battery at new position
        pp.create_storage(
            self.net, 
            bus=self.battery_position,
            p_mw=0,  # Will be controlled by SOC
            max_e_mwh=self.battery_capacity / 1000,
            soc_percent=self.battery_soc * 100,
            name="Battery"
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        try:
            pp.runpp(self.net, verbose=False)
            voltages = self.net.res_bus.vm_pu.values
        except:
            voltages = np.ones(self.n_buses)
        
        # Get current loads and PV generation
        time_idx = self.current_step % self.time_steps_per_episode
        loads = self.load_profiles[:, time_idx]
        pv_gen = self.pv_profiles[:, time_idx]
        
        # Normalize time of day
        time_of_day = (self.current_step % self.time_steps_per_episode) / self.time_steps_per_episode
        
        # Combine all state information
        state = np.concatenate([
            voltages,
            loads,
            pv_gen,
            [self.battery_soc],
            [time_of_day]
        ])
        
        return state.astype(np.float32)
    
    def _calculate_reward(self, converged: bool) -> float:
        """Calculate reward based on multiple objectives"""
        if not converged:
            return -1000  # Heavy penalty for non-convergence
        
        # Voltage regulation reward
        voltage_reward = self._voltage_regulation_reward()
        
        # Power loss minimization reward
        loss_reward = self._power_loss_reward()
        
        # Battery utilization reward
        battery_reward = self._battery_utilization_reward()
        
        # Solar integration reward
        solar_reward = self._solar_integration_reward()
        
        # Weighted combination
        total_reward = (
            0.4 * voltage_reward +
            0.3 * loss_reward +
            0.2 * battery_reward +
            0.1 * solar_reward
        )
        
        return total_reward
    
    def _voltage_regulation_reward(self) -> float:
        """Reward for maintaining voltages within limits"""
        voltages = self.net.res_bus.vm_pu.values
        violations = 0
        
        for v in voltages:
            if v < self.voltage_limits[0] or v > self.voltage_limits[1]:
                violations += abs(v - np.clip(v, self.voltage_limits[0], self.voltage_limits[1]))
        
        return -violations * 100  # Penalty for violations
    
    def _power_loss_reward(self) -> float:
        """Reward for minimizing power losses"""
        try:
            total_losses = self.net.res_line.pl_mw.sum()
            return -total_losses * 1000  # Convert to penalty
        except:
            return -10
    
    def _battery_utilization_reward(self) -> float:
        """Reward for effective battery utilization"""
        # Encourage keeping SOC in middle range for flexibility
        soc_penalty = abs(self.battery_soc - 0.5) * 10
        return -soc_penalty
    
    def _solar_integration_reward(self) -> float:
        """Reward for better solar integration"""
        time_idx = self.current_step % self.time_steps_per_episode
        total_pv = self.pv_profiles[:, time_idx].sum()
        
        # Encourage charging during high PV generation
        if total_pv > 0.5 and self.battery_soc < 0.8:
            return 5
        elif total_pv < 0.2 and self.battery_soc > 0.2:
            return 5
        else:
            return 0
    
    def _count_voltage_violations(self) -> int:
        """Count number of voltage violations"""
        try:
            voltages = self.net.res_bus.vm_pu.values
            violations = np.sum((voltages < self.voltage_limits[0]) | 
                              (voltages > self.voltage_limits[1]))
            return violations
        except:
            return self.n_buses
    
    def _calculate_power_losses(self) -> float:
        """Calculate total power losses"""
        try:
            return self.net.res_line.pl_mw.sum()
        except:
            return 999.0
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Battery Position: {self.battery_position}")
            print(f"Battery SOC: {self.battery_soc:.2f}")
            print(f"Voltage Violations: {self._count_voltage_violations()}")
            print(f"Power Losses: {self._calculate_power_losses():.4f} MW")
            print("-" * 40)
