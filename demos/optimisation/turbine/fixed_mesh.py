import argparse
import matplotlib.pyplot as plt
import numpy as np
import os

from firedrake.pyplot import tricontourf
from firedrake.utility_meshes import RectangleMesh

from goalie.adjoint import AdjointMeshSeq
from goalie.log import pyrint
from goalie.metric import ramp_complexity
from goalie.plot import plot_indicator_snapshots
from goalie.optimisation import QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.time_partition import TimeInstant

from setup import *

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot progress of controls and QoIs.")
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument("--taylor_test", action="store_true", help="Run a Taylor test.")
parser.add_argument("--plot_setup", action="store_true", help="Plot the problem setup.")
args = parser.parse_args()

# Use parsed arguments
n = args.n

experiment_id = f"fixed_mesh_{n}"
plot_dir = f"plots/{experiment_id}"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Set up the AdjointMeshSeq
mesh_seq = AdjointMeshSeq(
    TimeInstant(fields),
    RectangleMesh(60 * 2**n, 25 * 2**n, 1200, 500),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

# Plot the problem setup
if args.plot_setup:
    plot_setup("plots/setup.jpg")

# Run a Taylor test to check the gradient is computed correctly
if args.taylor_test:
    mesh_seq.taylor_test("yc")

# Solve the adjoint problem, computing gradients, and plot the x-velocity component of
# both the forward and adjoint solutions
solutions = mesh_seq.solve_adjoint(compute_gradient=True)
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/fixed_mesh_{n}_forward_unoptimised.jpg", bbox_inches="tight")
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
u_star, eta_star = solutions["solution_2d"]["adjoint"][0][0].subfunctions
fig.colorbar(tricontourf(u_star.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/fixed_mesh_{n}_adjoint_unoptimised.jpg", bbox_inches="tight")

J = mesh_seq.J
print(f"J = {J:.4e}")

# Set optimiser parameters, including a large starting step length
parameters = OptimisationParameters({"lr": 10.0})
print(parameters)

# Run the optimiser
optimiser = QoIOptimiser(mesh_seq, "yc", parameters, method="gradient_descent")
solutions = optimiser.minimise()

# Write the optimiser progress to file
output_dir = f"outputs/{experiment_id}"
np.save(f"{output_dir}/controls.npy", optimiser.progress["control"])
np.save(f"{output_dir}/qois.npy", optimiser.progress["qoi"])

# Plot the patches for the final positions
plot_patches(mesh_seq, optimiser.progress["control"][-1], f"{plot_dir}/patches.jpg")

# Plot the x-velocity component of the forward solution for the final control
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/fixed_mesh_{n}_forward_optimised.jpg", bbox_inches="tight")
