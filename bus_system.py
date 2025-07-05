import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as pt
import matplotlib.pyplot as plt

# Build or load your network
net = pn.case33bw()  # Example: IEEE 33-bus network

# Run power flow if you want voltage results
pp.runpp(net)

# Simple plot (shows buses, lines, loads, sgens, storage if present)
pt.simple_plot(net, plot_loads=True, plot_sgens=True)
plt.title("Network with Battery Placement")
plt.show()



def _create_distribution_network() -> pp.pandapowerNet:
        """Create a sample distribution network"""
        # Create IEEE 33-bus test system or custom network
        net = pn.case33bw()
        
        # Add PV systems at various buses
        pv_buses = [5, 10, 15, 20, 25, 30]
        for bus in pv_buses:
            pp.create_sgen(net, bus=bus, p_mw=0.1, q_mvar=0, 
                          name=f"PV_{bus}", type="PV")
        
        return net

dis_net = _create_distribution_network()
pp.runpp(dis_net)
pt.simple_plot(dis_net, plot_loads=True, plot_sgens=True)
plt.title("Network with Battery Placement")
plt.show()



import pandapower as pp
import matplotlib.pyplot as plt
import numpy as np

# Create empty network
net = pp.create_empty_network()

# Create buses
bus_mv = pp.create_bus(net, vn_kv=11, name="MV Bus 11kV")  # Medium voltage bus
bus_lv_main = pp.create_bus(net, vn_kv=0.4, name="LV Main Bus 400V")  # Main LV bus (400V side of transformer)

# Create external grid connection at MV bus
pp.create_ext_grid(net, bus_mv, vm_pu=1.0, name="Grid Connection")

# Create transformer 11kV/400V
pp.create_transformer(net, bus_mv, bus_lv_main, std_type="0.25 MVA 20/0.4 kV", name="Distribution Transformer")

# Create 10 feeder nodes connected in series
feeder_buses = []
for i in range(10):
    bus = pp.create_bus(net, vn_kv=0.4, name=f"Node {i+1}")
    feeder_buses.append(bus)
    
    # Create loads at each node (typical residential loads: 5kW active, 1kVAr reactive)
    pp.create_load(net, bus, p_mw=0.005, q_mvar=0.001, name=f"Load Node {i+1}")

# Create lines connecting nodes in series
# First line: LV Main Bus (400V side) to Node 1
pp.create_line(net, bus_lv_main, feeder_buses[0], length_km=0.1, 
               std_type="NAYY 4x50 SE", name="Main to Node 1")

# Series connections: Node 1 → Node 2 → Node 3 → ... → Node 10
for i in range(9):  # 9 connections for 10 nodes
    pp.create_line(net, feeder_buses[i], feeder_buses[i+1], length_km=0.05, 
                   std_type="NAYY 4x50 SE", name=f"Node {i+1} to Node {i+2}")

# Add rooftop solar to first 5 nodes
for i in range(5):
    # Negative power indicates generation
    # Typical rooftop solar: 3kW capacity
    pp.create_sgen(net, feeder_buses[i], p_mw=-0.003, q_mvar=0, 
                   name=f"Rooftop Solar Node {i+1}", type="PV")

# Run power flow analysis
pp.runpp(net)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Network topology (linear feeder)
ax1 = axes[0, 0]

# Position buses in a line for feeder visualization
positions = {}
positions[bus_mv] = (0, 2)  # MV bus at top
positions[bus_lv_main] = (0, 1)  # LV main bus below MV

# Feeder nodes in a horizontal line
for i, bus in enumerate(feeder_buses):
    positions[bus] = (i + 1, 0)

# Plot buses
ax1.scatter(*positions[bus_mv], s=200, c='red', marker='s', label='MV Bus (11kV)', zorder=3)
ax1.scatter(*positions[bus_lv_main], s=150, c='blue', marker='o', label='LV Main (400V)', zorder=3)

# Plot feeder nodes
for i, bus in enumerate(feeder_buses):
    if i < 5:  # Nodes with solar
        ax1.scatter(*positions[bus], s=120, c='orange', marker='o', 
                   label='Node with Solar' if i == 0 else "", zorder=3)
    else:  # Nodes without solar
        ax1.scatter(*positions[bus], s=100, c='green', marker='o', 
                   label='Load Node' if i == 5 else "", zorder=3)

# Plot transformer connection
ax1.plot([positions[bus_mv][0], positions[bus_lv_main][0]], 
         [positions[bus_mv][1], positions[bus_lv_main][1]], 
         'r-', linewidth=4, label='Transformer', zorder=2)

# Plot feeder lines
# Main to Node 1
ax1.plot([positions[bus_lv_main][0], positions[feeder_buses[0]][0]], 
         [positions[bus_lv_main][1], positions[feeder_buses[0]][1]], 
         'k-', linewidth=2, zorder=1)

# Node to node connections
for i in range(9):
    ax1.plot([positions[feeder_buses[i]][0], positions[feeder_buses[i+1]][0]], 
             [positions[feeder_buses[i]][1], positions[feeder_buses[i+1]][1]], 
             'k-', linewidth=2, zorder=1)

# Add node labels
for i, bus in enumerate(feeder_buses):
    ax1.annotate(f'N{i+1}', positions[bus], xytext=(5, 5), 
                textcoords='offset points', fontsize=8)

ax1.set_title('Radial Distribution Feeder Topology')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xlabel('Distance along feeder')
ax1.set_ylabel('Voltage Level')

# 2. Voltage profile along the feeder
ax2 = axes[0, 1]
# Get voltages for feeder buses only
feeder_voltages = [net.res_bus.loc[bus, 'vm_pu'] for bus in feeder_buses]
node_numbers = list(range(1, 11))

ax2.plot(node_numbers, feeder_voltages, 'bo-', linewidth=2, markersize=8, label='Voltage Profile')
ax2.set_xlabel('Node Number')
ax2.set_ylabel('Voltage (p.u.)')
ax2.set_title('Voltage Profile Along Feeder')
ax2.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Min Voltage Limit')
ax2.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Max Voltage Limit')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(node_numbers)

# 3. Line loading
ax3 = axes[1, 0]
line_loadings = net.res_line['loading_percent'].values
line_names = ['Main-N1'] + [f'N{i}-N{i+1}' for i in range(1, 10)]
ax3.bar(range(len(line_loadings)), line_loadings, color='skyblue', alpha=0.7)
ax3.set_xlabel('Line Segment')
ax3.set_ylabel('Loading (%)')
ax3.set_title('Line Loading Along Feeder')
ax3.set_xticks(range(len(line_names)))
ax3.set_xticklabels(line_names, rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# 4. Power flow at each node
ax4 = axes[1, 1]
load_powers = [net.res_load.loc[i, 'p_mw'] * 1000 for i in range(10)]  # Convert to kW
solar_powers = [0] * 10
for i in range(5):  # First 5 nodes have solar
    solar_powers[i] = net.res_sgen.loc[i, 'p_mw'] * 1000  # Convert to kW

x_pos = np.arange(1, 11)
width = 0.35
ax4.bar(x_pos - width/2, load_powers, width, label='Load (kW)', color='red', alpha=0.7)
ax4.bar(x_pos + width/2, solar_powers, width, label='Solar (kW)', color='orange', alpha=0.7)

ax4.set_xlabel('Node Number')
ax4.set_ylabel('Power (kW)')
ax4.set_title('Load and Generation at Each Node')
ax4.set_xticks(x_pos)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print detailed results
print("=== Radial Distribution Feeder Analysis ===")
print(f"Network converged: {net.converged}")
print(f"Total buses: {len(net.bus)}")
print(f"Total lines: {len(net.line)}")
print(f"Total loads: {len(net.load)}")
print(f"Total solar installations: {len(net.sgen)}")

print("\n=== Voltage Profile Along Feeder ===")
print("Node\tVoltage (p.u.)\tVoltage (V)")
for i, bus in enumerate(feeder_buses):
    voltage_pu = net.res_bus.loc[bus, 'vm_pu']
    voltage_v = voltage_pu * 400  # Convert to actual voltage
    print(f"N{i+1}\t{voltage_pu:.4f}\t\t{voltage_v:.1f}")

print("\n=== Line Loading Along Feeder ===")
line_names = ['Main-N1'] + [f'N{i}-N{i+1}' for i in range(1, 10)]
for i, (name, loading) in enumerate(zip(line_names, net.res_line['loading_percent'])):
    print(f"{name}:\t{loading:.2f}%")

print("\n=== Power Summary ===")
total_load = sum(net.res_load['p_mw']) * 1000  # kW
total_solar = sum(net.res_sgen['p_mw']) * 1000 if len(net.res_sgen) > 0 else 0  # kW
net_load = total_load + total_solar  # Solar is negative
print(f"Total Load: {total_load:.1f} kW")
print(f"Total Solar: {abs(total_solar):.1f} kW")
print(f"Net Load: {net_load:.1f} kW")

# Voltage drop analysis
voltage_drop = net.res_bus.loc[feeder_buses[0], 'vm_pu'] - net.res_bus.loc[feeder_buses[-1], 'vm_pu']
print(f"\nVoltage drop from Node 1 to Node 10: {voltage_drop:.4f} p.u. ({voltage_drop*400:.1f} V)")





##########################################################


import pandapower as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pandapower.plotting import create_bus_collection, create_line_collection, draw_collections
from pandapower.plotting.plotly import simple_plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# [Network creation code should be inserted here, lines 0-103]
import pandapower as pp
import pandapower.networks as pn
import pandapower.plotting as pt
import matplotlib.pyplot as plt

# Build or load your network
net = pn.case33bw()  # Example: IEEE 33-bus network

# Run power flow if you want voltage results
pp.runpp(net)

# Simple plot (shows buses, lines, loads, sgens, storage if present)
pt.simple_plot(net, plot_loads=True, plot_sgens=True)
plt.title("Network with Battery Placement")
plt.show()



def _create_distribution_network() -> pp.pandapowerNet:
        """Create a sample distribution network"""
        # Create IEEE 33-bus test system or custom network
        net = pn.case33bw()
        
        # Add PV systems at various buses
        pv_buses = [5, 10, 15, 20, 25, 30]
        for bus in pv_buses:
            pp.create_sgen(net, bus=bus, p_mw=0.1, q_mvar=0, 
                          name=f"PV_{bus}", type="PV")
        
        return net

dis_net = _create_distribution_network()
pp.runpp(dis_net)
pt.simple_plot(dis_net, plot_loads=True, plot_sgens=True)
plt.title("Network with Battery Placement")
plt.show()



import pandapower as pp
import matplotlib.pyplot as plt
import numpy as np

# Create empty network
net = pp.create_empty_network()

# Create buses
bus_mv = pp.create_bus(net, vn_kv=11, name="MV Bus 11kV")  # Medium voltage bus
bus_lv_main = pp.create_bus(net, vn_kv=0.4, name="LV Main Bus 400V")  # Main LV bus (400V side of transformer)

# Create external grid connection at MV bus
pp.create_ext_grid(net, bus_mv, vm_pu=1.0, name="Grid Connection")

# Create transformer 11kV/400V
pp.create_transformer(net, bus_mv, bus_lv_main, std_type="0.25 MVA 20/0.4 kV", name="Distribution Transformer")

# Create 10 feeder nodes connected in series
feeder_buses = []
for i in range(10):
    bus = pp.create_bus(net, vn_kv=0.4, name=f"Node {i+1}")
    feeder_buses.append(bus)
    
    # Create loads at each node (typical residential loads: 5kW active, 1kVAr reactive)
    pp.create_load(net, bus, p_mw=0.005, q_mvar=0.001, name=f"Load Node {i+1}")

# Create lines connecting nodes in series
# First line: LV Main Bus (400V side) to Node 1
pp.create_line(net, bus_lv_main, feeder_buses[0], length_km=0.1, 
               std_type="NAYY 4x50 SE", name="Main to Node 1")

# Series connections: Node 1 → Node 2 → Node 3 → ... → Node 10
for i in range(9):  # 9 connections for 10 nodes
    pp.create_line(net, feeder_buses[i], feeder_buses[i+1], length_km=0.05, 
                   std_type="NAYY 4x50 SE", name=f"Node {i+1} to Node {i+2}")

# Add rooftop solar to first 5 nodes
for i in range(5):
    # Negative power indicates generation
    # Typical rooftop solar: 3kW capacity
    pp.create_sgen(net, feeder_buses[i], p_mw=-0.003, q_mvar=0, 
                   name=f"Rooftop Solar Node {i+1}", type="PV")

# Run power flow analysis
pp.runpp(net)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Network topology (linear feeder)
ax1 = axes[0, 0]

# Position buses in a line for feeder visualization
positions = {}
positions[bus_mv] = (0, 2)  # MV bus at top
positions[bus_lv_main] = (0, 1)  # LV main bus below MV

# Feeder nodes in a horizontal line
for i, bus in enumerate(feeder_buses):
    positions[bus] = (i + 1, 0)

# Plot buses
ax1.scatter(*positions[bus_mv], s=200, c='red', marker='s', label='MV Bus (11kV)', zorder=3)
ax1.scatter(*positions[bus_lv_main], s=150, c='blue', marker='o', label='LV Main (400V)', zorder=3)

# Enhanced Voltage Visualization Functions
def create_voltage_heatmap_matplotlib(net):
    """Create voltage heatmap using matplotlib"""
    try:
        # Run power flow if not already done
        pp.runpp(net)

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Get voltage data
        voltages = net.res_bus['vm_pu'].values
        bus_names = net.bus['name'].values

        # Plot 1: Bus voltage heatmap
        voltage_colors = cm.RdYlGn(voltages)  # Red-Yellow-Green colormap
        bars1 = ax1.bar(range(len(voltages)), voltages, color=voltage_colors)
        ax1.set_xlabel('Bus Index')
        ax1.set_ylabel('Voltage (p.u.)')
        ax1.set_title('Bus Voltage Levels Heatmap')
        ax1.set_xticks(range(len(bus_names)))
        ax1.set_xticklabels(bus_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Min Limit (0.95)')
        ax1.axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='Max Limit (1.05)')
        ax1.legend()

        # Add voltage values on bars
        for i, (bar, voltage) in enumerate(zip(bars1, voltages)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{voltage:.3f}', ha='center', va='bottom', fontsize=8)

        # Plot 2: Line loading heatmap
        if len(net.res_line) > 0:
            line_loadings = net.res_line['loading_percent'].values
            line_names = net.line['name'].values
            loading_colors = cm.Reds(line_loadings/100)  # Red colormap based on loading

            bars2 = ax2.bar(range(len(line_loadings)), line_loadings, color=loading_colors)
            ax2.set_xlabel('Line Index')
            ax2.set_ylabel('Loading (%)')
            ax2.set_title('Line Loading Heatmap')
            ax2.set_xticks(range(len(line_names)))
            ax2.set_xticklabels(line_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Loading')
            ax2.legend()

            # Add loading values on bars
            for i, (bar, loading) in enumerate(zip(bars2, line_loadings)):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{loading:.1f}%', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.show()

        # Create colorbar reference
        fig_cb, ax_cb = plt.subplots(figsize=(8, 1))
        gradient = np.linspace(0.9, 1.1, 256).reshape(1, -1)
        ax_cb.imshow(gradient, aspect='auto', cmap='RdYlGn')
        ax_cb.set_xlim(0, 255)
        ax_cb.set_yticks([])
        ax_cb.set_xlabel('Voltage (p.u.)')
        ax_cb.set_title('Voltage Level Color Scale')

        # Set custom x-tick labels
        tick_positions = [0, 64, 128, 192, 255]
        tick_labels = ['0.90', '0.95', '1.00', '1.05', '1.10']
        ax_cb.set_xticks(tick_positions)
        ax_cb.set_xticklabels(tick_labels)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Matplotlib heatmap creation failed: {e}")

def create_voltage_heatmap_plotly(net):
    """Create interactive voltage heatmap using Plotly"""
    try:
        # Run power flow if not already done
        pp.runpp(net)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Bus Voltages', 'Line Loadings', 'Voltage Distribution', 'Power Flow'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Bus voltage heatmap
        voltages = net.res_bus['vm_pu'].values
        bus_names = net.bus['name'].values

        fig.add_trace(
            go.Bar(
                x=bus_names,
                y=voltages,
                marker=dict(
                    color=voltages,
                    colorscale='RdYlGn',
                    cmin=0.95,
                    cmax=1.05,
                    colorbar=dict(title="Voltage (p.u.)", x=0.45)
                ),
                text=[f'{v:.3f}' for v in voltages],
                textposition='outside',
                name='Bus Voltages'
            ),
            row=1, col=1
        )

        # Line loading heatmap
        if len(net.res_line) > 0:
            line_loadings = net.res_line['loading_percent'].values
            line_names = net.line['name'].values

            fig.add_trace(
                go.Bar(
                    x=line_names,
                    y=line_loadings,
                    marker=dict(
                        color=line_loadings,
                        colorscale='Reds',
                        colorbar=dict(title="Loading (%)", x=1.02)
                    ),
                    text=[f'{l:.1f}%' for l in line_loadings],
                    textposition='outside',
                    name='Line Loadings'
                ),
                row=1, col=2
            )

        # Voltage distribution histogram
        fig.add_trace(
            go.Histogram(
                x=voltages,
                nbinsx=20,
                marker=dict(color='lightblue', line=dict(color='black', width=1)),
                name='Voltage Distribution'
            ),
            row=2, col=1
        )

        # Power flow summary
        total_load = net.res_load['p_mw'].sum() * 1000  # Convert to kW
        total_solar = net.res_sgen['p_mw'].sum() * 1000  # Convert to kW
        net_power = total_solar - total_load

        fig.add_trace(
            go.Bar(
                x=['Total Load', 'Total Solar', 'Net Export'],
                y=[total_load, total_solar, net_power],
                marker=dict(
                    color=['red', 'green', 'blue' if net_power > 0 else 'orange'],
                ),
                text=[f'{total_load:.1f} kW', f'{total_solar:.1f} kW', f'{net_power:.1f} kW'],
                textposition='outside',
                name='Power Summary'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Distribution Feeder Voltage and Loading Analysis",
            showlegend=False,
            height=800
        )

        # Add voltage limit lines
        fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                      annotation_text="Min Voltage Limit", row=1, col=1)
        fig.add_hline(y=1.05, line_dash="dash", line_color="red",
                      annotation_text="Max Voltage Limit", row=1, col=1)

        fig.show()

    except Exception as e:
        print(f"Plotly heatmap creation failed: {e}")

def create_network_topology_heatmap(net):
    """Create network topology with voltage-colored buses and lines"""
    try:
        # Run power flow
        pp.runpp(net)

        # Create a simple network plot with voltage coloring
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create bus positions
        bus_coords = {}

        # HV bus position
        bus_coords[0] = (0, 2)  # HV bus
        bus_coords[1] = (2, 2)  # LV main bus

        # Feeder buses in a line
        for i in range(10):
            bus_coords[i+2] = (4 + i*0.8, 2 - 0.3*i)

        # Create collections for plotting
        voltages = net.res_bus['vm_pu'].values

        # Plot buses with voltage coloring
        for bus_idx, (x, y) in bus_coords.items():
            voltage = voltages[bus_idx]
            color = cm.RdYlGn(voltage) if 0.9 <= voltage <= 1.1 else 'black'
            size = 200 if bus_idx < 2 else 100  # Larger for HV and main LV bus

            ax.scatter(x, y, c=[voltage], cmap='RdYlGn', vmin=0.95, vmax=1.05,
                      s=size, edgecolors='black', linewidth=2)
            ax.annotate(f'{net.bus.loc[bus_idx, "name"]}\n{voltage:.3f} pu',
                       (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left')

        # Plot lines
        for _, line in net.line.iterrows():
            from_bus = line['from_bus']
            to_bus = line['to_bus']
            loading = net.res_line.loc[line.name, 'loading_percent']

            x_coords = [bus_coords[from_bus][0], bus_coords[to_bus][0]]
            y_coords = [bus_coords[from_bus][1], bus_coords[to_bus][1]]

            # Line color based on loading
            line_color = cm.Reds(loading/100) if loading <= 100 else 'red'
            line_width = 2 + (loading/50)  # Thicker lines for higher loading

            ax.plot(x_coords, y_coords, color=line_color, linewidth=line_width, alpha=0.7)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=0.95, vmax=1.05))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Voltage (p.u.)', rotation=270, labelpad=15)

        ax.set_title('Distribution Feeder Network - Voltage Heatmap')
        ax.set_xlabel('Distance')
        ax.set_ylabel('Distance')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Network topology heatmap creation failed: {e}")

# Function calls and analysis
print("\n" + "="*50)
print("VOLTAGE HEATMAP VISUALIZATIONS")
print("="*50)

# Create matplotlib heatmaps
print("\n1. Creating Matplotlib Heatmaps...")
create_voltage_heatmap_matplotlib(net)

# Create plotly interactive heatmaps
print("\n2. Creating Interactive Plotly Heatmaps...")
create_voltage_heatmap_plotly(net)

# Create network topology heatmap
print("\n3. Creating Network Topology Heatmap...")
create_network_topology_heatmap(net)

# Print detailed voltage analysis
print("\n" + "="*50)
print("DETAILED VOLTAGE ANALYSIS")
print("="*50)

voltages = net.res_bus['vm_pu']
print(f"Minimum voltage: {voltages.min():.4f} p.u. at bus {voltages.idxmin()}")
print(f"Maximum voltage: {voltages.max():.4f} p.u. at bus {voltages.idxmax()}")
print(f"Average voltage: {voltages.mean():.4f} p.u.")
print(f"Voltage standard deviation: {voltages.std():.4f} p.u.")

# Check voltage violations
low_voltage_buses = voltages[voltages < 0.95]
high_voltage_buses = voltages[voltages > 1.05]

if len(low_voltage_buses) > 0:
    print(f"\nLOW VOLTAGE VIOLATIONS ({len(low_voltage_buses)} buses):")
    for bus_idx, voltage in low_voltage_buses.items():
        print(f"  Bus {bus_idx} ({net.bus.loc[bus_idx, 'name']}): {voltage:.4f} p.u.")

if len(high_voltage_buses) > 0:
    print(f"\nHIGH VOLTAGE VIOLATIONS ({len(high_voltage_buses)} buses):")
    for bus_idx, voltage in high_voltage_buses.items():
        print(f"  Bus {bus_idx} ({net.bus.loc[bus_idx, 'name']}): {voltage:.4f} p.u.")
