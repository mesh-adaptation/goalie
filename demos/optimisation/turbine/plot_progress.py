import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot progress of controls and QoIs.")
parser.add_argument("--n", type=int, default=4, help="Initial mesh resolution.")
args = parser.parse_args()

# Use parsed arguments
n = args.n

# Define approaches
approaches = ["fixed_mesh", "goal_oriented_iso", "goal_oriented_aniso"]

# Plot controls for all approaches
fig, axes = plt.subplots()
for approach in approaches:
    try:
        controls = np.load(f"{approach}_{n}_control.npy")
        axes.plot(controls, "--x", label=approach)
    except FileNotFoundError:
        print(f"Control data for {approach} with n={n} not found.")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
axes.legend()
plt.savefig(f"controls_{n}.jpg", bbox_inches="tight")

# Plot QoIs for all approaches
fig, axes = plt.subplots()
for approach in approaches:
    try:
        qois = np.load(f"{approach}_{n}_qoi.npy")
        axes.plot(qois, "--x", label=approach)
    except FileNotFoundError:
        print(f"Control data for {approach} with n={n} not found.")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
axes.legend()
plt.savefig(f"qois_{n}.jpg", bbox_inches="tight")
