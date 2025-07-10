import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from setup import qoi_scaling
from utils import get_experiment_id

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on different axes."
)
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
args = parser.parse_args()
min_n = 0
max_n = 4
scaling = 1e-6 / qoi_scaling
experiment_id = get_experiment_id()
if not os.path.exists("plots"):
    os.makedirs("plots")

# Define approaches
go_labels = {
    "goal_oriented_iso": "Goal-oriented (isotropic)",
    "goal_oriented_aniso": "Goal-oriented (anisotropic)",
}

# Plot controls for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in range(min_n, max_n + 1):
    try:
        controls = np.load(f"outputs/fixed_mesh_{n}/control.npy")
        axes.plot(controls, "--x", label=f"Fixed mesh ({3000 * 2**n} elements)")
    except FileNotFoundError:
        print(f"Control data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for approach, label in go_labels.items():
    try:
        controls = np.load(f"outputs/{approach}_{n}_{experiment_id}/control.npy")
        axes.plot(controls, "--x", label=label)
    except FileNotFoundError:
        print(f"Control data for {approach} with n={n} not found.")
axes.set_xlabel("Iteration")
axes.set_ylabel(r"Control turbine position [$\mathrm{m}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"plots/controls.jpg", bbox_inches="tight")

# Plot QoIs for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in range(min_n, max_n + 1):
    try:
        qois = -np.load(f"outputs/fixed_mesh_{n}/qoi.npy") * scaling
        axes.plot(qois, "--x", label=f"Fixed mesh ({3000 * 2**n} elements)")
    except FileNotFoundError:
        print(f"QoI data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for approach, label in go_labels.items():
    try:
        qois = -np.load(f"outputs/{approach}_{n}_{experiment_id}/qoi.npy") * scaling
        axes.plot(qois, "--x", label=label)
    except FileNotFoundError:
        print(f"QoI data for {approach} with n={n} not found.")

axes.set_xlabel("Iteration")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"plots/qois.jpg", bbox_inches="tight")
