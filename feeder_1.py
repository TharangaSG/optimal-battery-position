import pandapower as pp
import numpy as np

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
                     std_type="0.25 MVA 10/0.4 kV",  # Standard transformer type
                     name="11kV/400V Transformer")

# Create feeder lines connecting main bus to each node
# Using typical LV cable specifications
line_lengths = [0.036, 0.02, 0.02, 0.025, 0.025, 0.026, 0.028, 0.023, 0.0225, 0.0225]  # km

for i, (bus, length) in enumerate(zip(feeder_buses, line_lengths)):
    if i == 0:
        # First node connects to main bus
        pp.create_line(net, 
                      from_bus=lv_main_bus, 
                      to_bus=bus, 
                      length_km=length,
                      std_type="NAYY 4x50 SE",  # Standard LV cable
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
# Typical Sri Lankan household loads: 2-5 kW
house_loads = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]  # kW
load_power_factors = [0.9] * 10  # Typical residential power factor

for i, (bus, p_load, pf) in enumerate(zip(feeder_buses, house_loads, load_power_factors)):
    q_load = p_load * np.tan(np.arccos(pf))  # Calculate reactive power
    pp.create_load(net, 
                   bus=bus, 
                   p_mw=p_load/1000,  # Convert kW to MW
                   q_mvar=q_load/1000,  # Convert kVAR to MVAR
                   name=f"House_Load_{i+1}")

# Create rooftop solar systems 
solar_nodes = [4, 5, 6, 7]  # Indices for poldes 5, 6, 7, 8  
solar_capacities = [17, 5.0, 16, 3.3]  # kW - typical rooftop solar sizes (10+6 for pole 7)

for i, (node_idx, solar_kw) in enumerate(zip(solar_nodes, solar_capacities)):
    pp.create_sgen(net, 
                   bus=feeder_buses[node_idx], 
                   p_mw=solar_kw/1000,  # Convert kW to MW
                   q_mvar=0,  # Assume unity power factor for solar
                   name=f"Rooftop_Solar_{node_idx+1}",
                   type="PV")


# # Add SECOND solar system to Pole8
pp.create_sgen(net, 
               bus=feeder_buses[6],  # pole8
               p_mw=2.5/1000,  # 2.5 kW additional solar system
               q_mvar=0,  # Unity power factor
               name="Solar_System_Node3_Additional")

# Print network summary
print("=== Distribution Feeder Network Summary ===")
print(f"Total Buses: {len(net.bus)}")
print(f"Total Lines: {len(net.line)}")
print(f"Total Transformers: {len(net.trafo)}")
print(f"Total Loads: {len(net.load)}")
print(f"Total Solar Generators: {len(net.sgen)}")
print(f"Total Load: {sum(house_loads):.1f} kW")
print(f"Total Solar Capacity: {sum(solar_capacities):.1f} kW")

# Display network tables
print("\n=== Bus Information ===")
print(net.bus)

print("\n=== Load Information ===")
print(net.load)

print("\n=== Solar Generator Information ===")
print(net.sgen)

# Run power flow analysis
try:
    pp.runpp(net)
    print("\n=== Power Flow Results ===")
    print("Bus Voltages (p.u.):")
    print(net.res_bus[['vm_pu', 'va_degree']])
    
    print("\nLine Loading:")
    print(net.res_line[['loading_percent', 'p_from_mw', 'q_from_mvar']])
    
    print("\nTransformer Loading:")
    print(net.res_trafo[['loading_percent', 'p_hv_mw', 'q_hv_mvar']])
    
except Exception as e:
    print(f"Power flow calculation failed: {e}")

# Optional: Create a simple network plot (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    from pandapower.plotting import simple_plot
    
    # Create a simple plot
    simple_plot(net, show_plot=True, plot_loads=True, plot_sgens=True)
    
    
except ImportError:
    print("something whent wrong.")


# View all static generators (solar systems)
print("Solar systems in the network:")
print(net.sgen)

# Check specifically for Node 3 (bus index 2)
node3_solars = net.sgen[net.sgen.bus == 6]  # feeder_buses[2] is Node 3
print(f"\nSolar systems at Node 3: {len(node3_solars)}")
print(node3_solars)

