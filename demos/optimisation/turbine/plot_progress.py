import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from setup import qoi_scaling
from utils import get_latest_experiment_id

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on different axes."
)
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--hash", type=str, default=None, help="Git hash identifier for the experiment."
)
# TODO: Accept multiple target complexities
args = parser.parse_args()
min_n = 0
max_n = 4
scaling = 1e-6 / qoi_scaling
experiment_id = get_latest_experiment_id(hash=args.hash)
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Define approaches
go_labels = {
    False: "Goal-oriented (isotropic)",
    True: "Goal-oriented (anisotropic)",
}

# Plot controls for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in range(min_n, max_n + 1):
    try:
        controls = np.load(f"outputs/fixed_mesh_{n}/controls.npy")
        axes.plot(controls, "--x", label=f"Fixed mesh ({3000 * 2**n} elements)")
    except FileNotFoundError:
        print(f"Control data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = f"goal_oriented_n{n}_anisotropic{int(anisotropic)}"
    try:
        controls = np.load(f"{output_dir}/{model_config}_controls.npy")
        axes.plot(controls, "--x", label=label)
    except FileNotFoundError:
        print(f"Control data not found for {model_config}.")
axes.set_xlabel("Iteration")
axes.set_ylabel(r"Control turbine position [$\mathrm{m}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/controls.jpg", bbox_inches="tight")

# Plot QoIs for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in range(min_n, max_n + 1):
    try:
        qois = -np.load(f"outputs/fixed_mesh_{n}/qois.npy") * scaling
        axes.plot(qois, "--x", label=f"Fixed mesh ({3000 * 2**n} elements)")
    except FileNotFoundError:
        print(f"QoI data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = f"goal_oriented_n{n}_anisotropic{int(anisotropic)}"
    try:
        qois = -np.load(f"{output_dir}/{model_config}_qois.npy") * scaling
        axes.plot(qois, "--x", label=label)
    except FileNotFoundError:
        print(f"QoI data not found for {model_config}.")

axes.set_xlabel("Iteration")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/qois.jpg", bbox_inches="tight")
