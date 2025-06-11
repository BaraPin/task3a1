import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
import csv

# Simulation parameters
p = ct.one_atm  # pressure [Pa]
Tin = 300.0     # unburned gas temperature [K]
phi = 0.75      # equivalence ratio
width = 0.009   # flame domain width in meters (9 mm)

# Create gas object
gas = ct.Solution("gri30.yaml")
gas.TP = Tin, p
gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")

# Set up flame
f = ct.FreeFlame(gas, width=width)
f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
f.transport_model = "default"
f.flame.set_steady_tolerances(default=[1e-9, 1e-5])
f.solve(loglevel=0, auto=True, refine_grid=True)

# Print flame speed
laminar_flame_speed = f.velocity[0]
print("\nFlame speed = {:.3f} m/s\n".format(laminar_flame_speed))

# Convert distance to mme   
x_mm = f.grid * 1000

# Plot
plt.rcParams["font.size"] = 14
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot temperature
line1, = ax1.plot(x_mm, f.T, label="Temperature (K)", color='red')  # ← důležité pro legendu
ax1.set_xlabel("Distance (mm)")
ax1.set_ylabel("Temperature (K)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Plot mole fractions
ax2 = ax1.twinx()
species = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'OH', 'H', 'H2']
colors = ['blue', 'green', 'purple', 'orange', 'black', 'magenta', 'gray', 'yellow']

for sp, color in zip(species, colors):
    y_data = f.X[gas.species_index(sp)]

    if sp in ['OH', 'H', 'H2']:
        y_data = 10 * y_data
        label = f"10×{sp}"
    else:
        label = sp

    ax2.plot(x_mm, y_data, label=label, color=color)


# ax2.plot(x_mm, f.X[gas.species_index(sp)], label=sp, color=color)

ax2.set_ylabel("Mole Fraction")
ax2.tick_params(axis='y')

# Combined legend
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend([line1] + lines2, ["Temperature (K)"] + labels2, loc='upper right')

plt.savefig("methane_flame_structure_a.png")


# import cantera as ct
# import numpy as np
# import matplotlib.pyplot as plt
# import csv

# # Simulation parameters
# p = ct.one_atm  # pressure [Pa]
# Tin = 300.0  # unburned gas temperature [K]
# phi = 1 
# width = 0.4

# # IdealGasMix object used to compute mixture properties 
# gas = ct.Solution("gri30.yaml")
# gas.TP = Tin, p
# gas.set_equivalence_ratio(phi, "CH4", "O2:1.0, N2:3.76")

# # Flame object
# f = ct.FreeFlame(gas, width=width)
# f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
# f.transport_model = "default"
# f.flame.set_steady_tolerances(default=[1e-9, 1e-5])
# f.solve(loglevel=0, auto=True, refine_grid=True)

# laminar_flame_speed = f.velocity[0]
# print("\nflamespeed = {:7f} m/s\n".format(laminar_flame_speed))

# # Plot temperature profile
# plt.rcParams["font.size"] = 14
# plt.rcParams["xtick.labelsize"] = 10
# plt.rcParams["ytick.labelsize"] = 10

# fig, ax1 = plt.subplots(figsize=(10, 6))

# # Left y-axis: Temperature
# ax1.plot(f.grid, f.T, color='red', label="Temperature (K)")
# ax1.set_xlabel("Distance (m)")
# ax1.set_ylabel("Temperature (K)", color='red')
# ax1.tick_params(axis='y', labelcolor='red')

# # Right y-axis: Mole fractions
# ax2 = ax1.twinx()
# species = ['CH4', 'O2', 'CO2', 'H2O', 'CO', 'H2O2']
# colors = ['blue', 'green', 'purple', 'orange', 'black', 'yellow']

# for sp, color in zip(species, colors):
#     ax2.plot(f.grid, f.X[gas.species_index(sp)], label=sp, color=color)

# ax2.set_ylabel("Mole Fraction")
# ax2.tick_params(axis='y')

# # Legends
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# ax1.grid(True)
# plt.title("Temperature and Species Mole Fractions")
# fig.tight_layout()

# plt.savefig("temperature_species_profile.png")
# plt.show()

# # Save temperature data
# data = np.vstack((f.grid, f.T)).T
# with open("flame_speed.csv", "w", newline="") as f_csv:
#     writer = csv.writer(f_csv)
#     writer.writerows(data)
