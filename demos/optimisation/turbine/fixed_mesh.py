import matplotlib.pyplot as plt
from firedrake.pyplot import tricontourf
from firedrake.utility_meshes import RectangleMesh
from setup import *

from goalie import *

# Set up the AdjointMeshSeq
n = 4
mesh_seq = AdjointMeshSeq(
    TimeInstant(fields),
    RectangleMesh(12 * n, 5 * n, 1200, 500),
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)

# # FIXME: Sub-quadratic convergence
# # Run a Taylor test to check the gradient is computed correctly
# mesh_seq.taylor_test("yc")

# Solve the adjoint problem, computing gradients, and plot the x-velocity component of
# both the forward and adjoint solutions
solutions = mesh_seq.solve_adjoint(compute_gradient=True)
u, eta = solutions["solution_2d"]["forward"][0][0].subfunctions
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Forward $x$-velocity")
fig.colorbar(tricontourf(u.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"fixed_mesh_{n}_forward.jpg", bbox_inches="tight")
fig, axes = plt.subplots(figsize=(12, 5))
axes.set_title(r"Adjoint $x$-velocity")
u_star, eta_star = solutions["solution_2d"]["adjoint"][0][0].subfunctions
fig.colorbar(tricontourf(u_star.subfunctions[0], axes=axes, cmap="coolwarm"), ax=axes)
plt.savefig(f"fixed_mesh_{n}_adjoint.jpg", bbox_inches="tight")

J = mesh_seq.J
print(f"J = {J:.4e}")

parameters = OptimisationParameters()
print(parameters)

optimiser = QoIOptimiser(mesh_seq, "yc", parameters, method="gradient_descent")
optimiser.minimise()

fig, axes = plt.subplots()
axes.plot(optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig(f"fixed_mesh_{n}_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig(f"fixed_mesh_{n}_qoi.jpg", bbox_inches="tight")
