import argparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

from animate.metric import RiemannianMetric
from animate.adapt import adapt
from firedrake.checkpointing import CheckpointFile
from firedrake.function import Function
from firedrake.functionspace import TensorFunctionSpace
from firedrake.pyplot import tricontourf
from firedrake.utility_meshes import RectangleMesh

from goalie.go_mesh_seq import GoalOrientedMeshSeq
from goalie.log import pyrint
from goalie.metric import ramp_complexity
from goalie.plot import plot_indicator_snapshots
from goalie.optimisation import QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.time_partition import TimeInstant

from setup import *
from utils import get_experiment_id

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Run with goal-oriented adaptation.")
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--anisotropic",
    action="store_true",
    help="Use anisotropic adaptation (default: False).",
)
args, _ = parser.parse_known_args()

# Use parsed arguments
n = args.n
anisotropic = args.anisotropic
config_str = f"goal_oriented_n{n}_anisotropic{int(anisotropic)}"

# Determine the experiment_id and create associated directories
experiment_id = get_experiment_id()
print(f"Experiment ID: {experiment_id}")
plot_dir = os.path.join("plots", experiment_id)
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
output_dir = os.path.join("outputs", experiment_id)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
data_dir = os.path.join("data", experiment_id)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Set up the GoalOrientedMeshSeq
mesh_seq = GoalOrientedMeshSeq(
    TimeInstant(fields),
    RectangleMesh(int(np.round(60 * 2**n)), int(np.round(25 * 2**n)), 1200, 500),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

# Solve the adjoint problem, computing gradients, and plot the x-velocity component of
# both the forward and adjoint solutions
solutions = mesh_seq.solve_adjoint(compute_gradient=True)
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Forward $x$-velocity")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/{config_str}_forward_unoptimised.jpg", bbox_inches="tight")
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Adjoint $x$-velocity")
u_star, eta_star = solutions["solution_2d"]["adjoint"][0][0].subfunctions
fig.colorbar(tricontourf(u_star.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/{config_str}_adjoint_unoptimised.jpg", bbox_inches="tight")

J = mesh_seq.J
print(f"J = {J:.4e}")

# Set optimiser parameters, including a large starting step length
parameters = OptimisationParameters({"lr": 10.0})
print(parameters)


def adaptor(mesh_seq, solutions, indicators):
    P1_ten = TensorFunctionSpace(mesh_seq[0], "CG", 1)
    metric = RiemannianMetric(P1_ten)

    # Ramp the target metric complexity over the first few iterations
    base = 1000
    target = 1000  # FIXME: Avoid adjoint solver fail with larger values
    iteration = mesh_seq.fp_iteration
    num_iterations = 3
    mp = {
        "dm_plex_metric_target_complexity": ramp_complexity(
            base, target, iteration, num_iterations=num_iterations
        ),
        "dm_plex_metric_hausdorff_number": 0.01 * 1000,
    }
    metric.set_parameters(mp)

    if anisotropic:
        # Recover the Hessian of the forward solution
        hessians = {key: RiemannianMetric(P1_ten) for key in ("u", "v", "eta")}
        uv, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
        hessians["u"].compute_hessian(uv[0])
        hessians["v"].compute_hessian(uv[1])
        hessians["eta"].compute_hessian(eta)
        # FIXME: Why doesn't intersection work here?
        hessian = hessians["u"].average(hessians["v"], hessians["eta"])

        # Deduce an anisotropic metric from the error indicator field and the Hessian
        metric.compute_anisotropic_dwr_metric(indicators["solution_2d"][0][0], hessian)
        # FIXME: Why does it only refine the inflow?
        # TODO: Plot Hessian components for debugging (indicators seem fine)
    else:
        # Deduce an isotropic metric from the error indicator field
        metric.compute_isotropic_dwr_metric(indicators["solution_2d"][0][0])
    complexity = metric.complexity()

    # Adapt the mesh
    mesh_seq[0] = adapt(mesh_seq[0], metric)
    num_dofs = mesh_seq.count_vertices()[0]
    num_elem = mesh_seq.count_elements()[0]
    pyrint(
        f"{iteration:2d}, complexity: {complexity:4.0f}"
        f", dofs: {num_dofs:4d}, elements: {num_elem:4d}"
    )

    # Write each intermediate adapted mesh
    checkpoint_filename = os.path.join(data_dir, f"{config_str}_mesh{iteration}.h5")
    with CheckpointFile(checkpoint_filename, "w") as chk:
        chk.save_mesh(mesh_seq[0])

    # Plot error indicator on intermediate meshes
    plot_kwargs = {"figsize": (12, 5)}
    plot_kwargs["norm"] = mcolors.LogNorm()
    plot_kwargs["locator"] = ticker.LogLocator()
    fig, axes, tcs = plot_indicator_snapshots(
        indicators, mesh_seq.time_partition, "solution_2d", **plot_kwargs
    )
    axes.set_title(f"Indicator at iteration {iteration}")
    fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
    fig.savefig(f"{plot_dir}/{config_str}_indicator{iteration}.jpg")
    plt.close()

    # Check whether the target complexity has been (approximately) reached. If not,
    # return ``True`` to indicate that convergence checks should be skipped until the
    # next fixed point iteration.
    continue_unconditionally = complexity < 0.95 * target
    return [continue_unconditionally]


# Run the optimiser
optimiser = QoIOptimiser(
    mesh_seq, "yc", parameters, method="gradient_descent", adaptor=adaptor
)
optimiser.minimise(dropout=False)

# Write the optimiser progress to file
np.save(f"{output_dir}/{config_str}_controls.npy", optimiser.progress["control"])
np.save(f"{output_dir}/{config_str}_qois.npy", optimiser.progress["qoi"])

# Plot the patches for the final positions
plot_patches(
    mesh_seq, optimiser.progress["control"][-1], f"{plot_dir}/{config_str}_patches.jpg"
)

# Plot the x-velocity component of the forward solution for the final control/mesh
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_xlabel(r"x-coordinate $\mathrm{[m]}$")
axes.set_ylabel(r"y-coordinate $\mathrm{[m]}$")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"{plot_dir}/{config_str}_forward_optimised.jpg", bbox_inches="tight")
