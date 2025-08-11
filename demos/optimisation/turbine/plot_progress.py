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
n_range = [0, 1, 2, 2.5850]
scaling = 1e-6 / qoi_scaling
experiment_id = get_latest_experiment_id(hash=args.hash)
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Define approaches
go_labels = {False: "Isotropic goal-oriented", True: "Anisotropic goal-oriented"}

# Plot controls for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in n_range:
    try:
        n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
        dofs = np.load(f"outputs/fixed_mesh_{n_str}/dofs.npy")[-1]
        timings = np.load(f"outputs/fixed_mesh_{n_str}/timings.npy")
        controls = np.load(f"outputs/fixed_mesh_{n_str}/controls.npy")
        axes.semilogx(timings, controls, "--x", label=f"Fixed mesh ({dofs:.0f} DoFs)")
    except FileNotFoundError:
        print(f"Control data for fixed_mesh with n={n} not found.")

# Plot goal-oriented cases
n = args.n
# TODO: Vary complexity
base = 1000
target = 1000

for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = (
        f"goal_oriented_n{n}_anisotropic{int(anisotropic)}_base{base}_target{target}"
    )
    try:
        timings = np.load(f"{output_dir}/{model_config}_timings.npy")
        controls = np.load(f"{output_dir}/{model_config}_controls.npy")
        axes.semilogx(timings, controls, "--x", label=label)
    except FileNotFoundError:
        print(f"Control data not found for {model_config}.")
axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
axes.set_ylabel(r"Control turbine position [$\mathrm{m}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/controls.jpg", bbox_inches="tight")

# Plot QoIs for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in n_range:
    try:
        n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
        dofs = np.load(f"outputs/fixed_mesh_{n_str}/dofs.npy")[-1]
        timings = np.load(f"outputs/fixed_mesh_{n_str}/timings.npy")
        qois = -np.load(f"outputs/fixed_mesh_{n_str}/qois.npy") * scaling
        axes.semilogx(timings, qois, "--x", label=f"Fixed mesh ({dofs:.0f} DoFs)")
    except FileNotFoundError:
        print(f"QoI data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = (
        f"goal_oriented_n{n}_anisotropic{int(anisotropic)}_base{base}_target{target}"
    )
    try:
        timings = np.load(f"{output_dir}/{model_config}_timings.npy")
        qois = -np.load(f"{output_dir}/{model_config}_qois.npy") * scaling
        axes.semilogx(timings, qois, "--x", label=label)
    except FileNotFoundError:
        print(f"QoI data not found for {model_config}.")

axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/qois.jpg", bbox_inches="tight")

# Plot gradients for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in n_range:
    try:
        n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
        dofs = np.load(f"outputs/fixed_mesh_{n_str}/dofs.npy")[-1]
        timings = np.load(f"outputs/fixed_mesh_{n_str}/timings.npy")
        gradients = np.abs(
            np.load(f"outputs/fixed_mesh_{n_str}/gradients.npy") * scaling
        )
        gradients /= gradients[0]  # Normalise by the first value
        axes.loglog(timings, gradients, "--x", label=f"Fixed mesh ({dofs:.0f} DoFs)")
    except FileNotFoundError:
        print(f"Gradient data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = (
        f"goal_oriented_n{n}_anisotropic{int(anisotropic)}_base{base}_target{target}"
    )
    try:
        timings = np.load(f"{output_dir}/{model_config}_timings.npy")
        gradients = np.abs(
            np.load(f"{output_dir}/{model_config}_gradients.npy") * scaling
        )
        gradients /= gradients[0]  # Normalise by the first value
        axes.loglog(timings, gradients, "--x", label=label)
    except FileNotFoundError:
        print(f"Gradient data not found for {model_config}.")

axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
axes.set_ylabel("Gradient relative to initial value")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/gradients.jpg", bbox_inches="tight")

# Plot dof counts for all approaches
fig, axes = plt.subplots()
# Plot fixed mesh cases
for n in n_range:
    try:
        n_str = n if np.isclose(n, np.round(n)) else f"{n:.4f}".replace(".", "p")
        dofs = np.load(f"outputs/fixed_mesh_{n_str}/dofs.npy")
        timings = np.load(f"outputs/fixed_mesh_{n_str}/timings.npy")
        axes.loglog(timings, dofs, "--x", label=f"Fixed mesh ({dofs[-1]:.0f} DoFs)")
    except FileNotFoundError:
        print(f"DoF count data for fixed_mesh with n={n} not found.")
# Plot goal-oriented cases
n = args.n
for anisotropic, label in go_labels.items():
    output_dir = f"outputs/{experiment_id}"
    model_config = (
        f"goal_oriented_n{n}_anisotropic{int(anisotropic)}_base{base}_target{target}"
    )
    try:
        dofs = np.load(f"{output_dir}/{model_config}_dofs.npy")
        timings = np.load(f"{output_dir}/{model_config}_timings.npy")
        axes.loglog(timings, dofs, "--x", label=label)
    except FileNotFoundError:
        print(f"DoF count data not found for {model_config}.")

axes.set_xlabel(r"CPU time [$\mathrm{s}$]")
axes.set_ylabel("DoF count")
axes.grid(True)
axes.legend()
plt.savefig(f"{plot_dir}/dofs.jpg", bbox_inches="tight")
