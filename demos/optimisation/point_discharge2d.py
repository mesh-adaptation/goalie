# Steady-state PDE-constrained optimisation
# =========================================

# Recall the steady-state advection-diffusion test case from :cite:`Riadh:2014`, in
# which we solve
#
# .. math::
#   \left\{\begin{array}{rl}
#       \mathbf u\cdot\nabla c - \nabla\cdot(D\nabla c) = S & \text{in}\:\Omega\\
#       c=0 & \text{on}\:\partial\Omega_{\mathrm{inflow}}\\
#       \nabla c\cdot\widehat{\mathbf n}=0 &
#           \text{on}\:\partial\Omega\backslash\partial\Omega_{\mathrm{inflow}}
#   \end{array}\right.,
#
# for a tracer concentration :math:`c`, with fluid velocity
# :math:`\mathbf u`, diffusion coefficient :math:`D` and point source
# representation :math:`S`. The domain of interest is the rectangle
# :math:`\Omega = [0, 50] \times [0, 10]`.
#
# As mentioned previously, point sources are difficult to represent in numerical models.
# In the `first point discharge demo <../point_discharge2d.py.html>`__, we used a
# formulation in terms of a radius parameter, :math:`r=0.05606388`. This value was
# deduced in a calibration experiment in :cite:`Wallwork:2022`, which we revisit here.
# The idea is to ensure that the finite element approximation is as close as possible
# to the analytical solution, in some sense.
#
# As always, start by importing Firedrake and Goalie. We also import some plotting
# utilities. ::

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import *

from goalie_adjoint import *

# For linear steady-state problems, we do not usually need to specify
# :func:`get_initial_condition` because the PDE is independent of any initial guess.
# However, we make use of several fields that aren't solved for in this problem and so
# do specify their values using the interface for setting initial conditions. We solve
# the advection-diffusion problem in :math:`\mathbb P1` space, while all other
# parameters are defined in Real space. ::

n = 2
mesh = RectangleMesh(100 * n, 20 * n, 50, 10)
fields = [
    Field("c", mesh=mesh, family="Lagrange", degree=1, unsteady=False),
    Field("u_x", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("u_y", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("D", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("r", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
]


def get_initial_condition(mesh_seq):
    P1 = mesh_seq.function_spaces["c"][0]
    R = mesh_seq.function_spaces["r"][0]
    return {
        "c": Function(P1),
        "u_x": Function(R).assign(1.0),
        "u_y": Function(R).assign(0.0),
        "D": Function(R).assign(0.1),
        "r": Function(R).assign(0.1),  # Radius parameter, to be calibrated
    }


# Copy over the solver from the original demo, dropping the :meth:`mesh_seq.read_form()`
# call, which is not relevant here because we aren't making use of
# :class:`~.GoalOrientedMeshSeq`. ::


def get_solver(mesh_seq):
    def solver(index):
        function_space = mesh_seq.function_spaces["c"][index]

        # Define constants
        c = mesh_seq.field_functions["c"]
        u_x = mesh_seq.field_functions["u_x"]
        u_y = mesh_seq.field_functions["u_y"]
        u = as_vector([u_x, u_y])
        D = mesh_seq.field_functions["D"]

        # Define source
        r = mesh_seq.field_functions["r"]
        x, y = SpatialCoordinate(mesh_seq[index])
        x0, y0 = 2, 5
        S = 100.0 * exp(-((x - x0) ** 2 + (y - y0) ** 2) / r**2)

        # SUPG stabilisation parameter
        h = CellSize(mesh_seq[index])
        unorm = sqrt(dot(u, u))
        tau = 0.5 * h / unorm
        tau = min_value(tau, unorm * h / (6 * D))

        # Setup variational problem
        psi = TestFunction(function_space)
        psi = psi + tau * dot(u, grad(psi))
        F = (
            dot(u, grad(c)) * psi * dx
            + inner(D * grad(c), grad(psi)) * dx
            - S * psi * dx
        )
        bc = DirichletBC(function_space, 0, 1)

        solve(F == 0, c, bcs=bc, ad_block_tag="c")
        yield

    return solver


# The analytical solution for this problem was presented in :cite:`Riadh:2014`. It
# includes a Bessel function, which tends towards infinity as its argument goes to zero.
# The value that gets passed as the argument is the mesh Peclet number multiplied by the
# square of the radial distance from the point source. As such, we introduce a
# thresholding for the radius parameter, r. ::


def analytical_solution(mesh_seq, index):
    """
    Analytical solution for point-discharge with diffusion problem.
    """
    u_x = mesh_seq.field_functions["u_x"]
    D = mesh_seq.field_functions["D"]
    r = mesh_seq.field_functions["r"]
    x, y = SpatialCoordinate(mesh_seq[index])
    x0, y0 = 2, 5  # Location of the point source
    Pe = 0.5 * u_x / D  # Mesh Peclet number

    # Define thresholding for the Bessel function argument
    r_thresh = max_value(sqrt((x - x0) ** 2 + (y - y0) ** 2), r**2)

    return 0.5 / (pi * D) * exp(Pe * (x - x0)) * bessk0(Pe * r_thresh)


# The QoI for the problem is defined as an error between the approximate solution and
# the analytical solution above. In particular, we use the :math:`L^2` error. It turns
# out that taking the :math:`L^2` error over the whole domain doesn't give particularly
# useful results because the analytical solution tends to infinity at the point source
# (as it should) but no approximate solution will have this property. As such, we use
# the same thresholding again such that we only consider the :math:`L^2` error outside
# of the disc of radius :math:`r` around the point source. ::


def get_qoi(mesh_seq, index):
    def qoi():
        c = mesh_seq.field_functions["c"]
        c_ana = analytical_solution(mesh_seq, index)

        # Define kernel for neglecting the disc around the point source
        r = mesh_seq.field_functions["r"]
        x0, y0 = 2, 5  # Location of the point source
        x, y = SpatialCoordinate(mesh_seq[index])
        kernel = conditional((x - x0) ** 2 + (y - y0) ** 2 > r**2, 1, 0)

        # L2 error scaled by kernel
        return kernel * (c - c_ana) ** 2 * dx

    return qoi


# Finally, we can set up the problem, which is the same as before. Solve the forward and
# adjoint problems with the initial configuration so we can get an idea of what the
# fields and parameters look like. ::

time_partition = TimeInstant(fields)
mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="steady",
)
solutions = mesh_seq.solve_adjoint(compute_gradient=True)

# Plot the solution fields and check the initial QoI and gradient. ::

plot_kwargs = {
    "levels": np.linspace(0, 1.65, 33),
    "figsize": (10, 3),
    "cmap": "coolwarm",
}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Uncalibrated forward solution")
fig.savefig("point_discharge2d-forward_uncalibrated.jpg")

# Piggy-pack off the existing data structure to conveniently plot the analytical
# solution. ::

solutions["c"]["forward"][0][0].interpolate(analytical_solution(mesh_seq, 0))
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Analytical solution")
fig.savefig("point_discharge2d-analytical.jpg")

# Plot the adjoint solution, too, which has a different scale. ::

plot_kwargs = {"figsize": (10, 3), "cmap": "coolwarm", "levels": 33}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "adjoint", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Adjoint solution")
fig.savefig("point_discharge2d-adjoint.jpg")

J = mesh_seq.J
print(f"Initial radius: {float(mesh_seq.field_functions["r"]):.8f}")
print(f"Initial QoI: {J:.4e}")
print(f"Initial gradient: {float(mesh_seq.gradient["r"]):.4e}")

# This should give output similar to the following:
#
# .. code-block:: none
#
#     Initial radius: 0.10000000
#     Initial QoI: 3.8960e+01
#     Initial gradient: 4.5085e+00
#
# Note that the gradient with respect to the intitial condition doesn't make sense in
# this example because we have a steady-state, linear PDE, wherein the solution doesn't
# depend on the initial condition.
#
# .. figure:: point_discharge2d-forward_uncalibrated.jpg
#    :figwidth: 80%
#    :align: center
#
# .. figure:: point_discharge2d-analytical.jpg
#    :figwidth: 80%
#    :align: center
#
# For both plots, the scale maxes out due to large values near to the point source. The
# uncalibrated approximate solution is clearly quite far from the analytical solution.
#
# .. figure:: point_discharge2d-adjoint.jpg
#    :figwidth: 80%
#    :align: center
#
# The adjoint solution looks much like we saw previously. This isn't surprising, given
# that it involves the same kernel function.
#
# Now run the optimisation routine and plot the results. ::

parameters = OptimisationParameters({"lr": 0.001, "maxiter": 100})
optimiser = QoIOptimiser(mesh_seq, "r", parameters, method="gradient_descent")
optimiser.minimise()

# This should give output similar to the following:
#
# .. code-block:: none
#
#     it= 0, r=1.0000e-01
#     it= 1, r=9.5492e-02, J=3.8960e+01, dJ=4.5085e+00, lr=1.0000e-03
#     it= 2, r=7.2096e-02, J=2.9998e+01, dJ=3.7801e+00, lr=6.1893e-03
#     it= 3, r=6.2308e-02, J=4.6765e+00, dJ=1.1149e+00, lr=8.7785e-03
#     it= 4, r=5.5342e-02, J=1.1478e+00, dJ=4.6359e-01, lr=1.5027e-02
#     it= 5, r=5.2727e-02, J=1.5863e-01, dJ=1.2655e-01, lr=2.0669e-02
#     it= 6, r=5.2156e-02, J=5.6844e-02, dJ=2.2659e-02, lr=2.5177e-02
#     it= 7, r=5.2115e-02, J=5.3308e-02, dJ=1.5268e-03, lr=2.6996e-02
#     it= 8, r=5.2114e-02, J=5.3316e-02, dJ=2.0780e-05, lr=2.7368e-02
#     Gradient convergence detected
#     Calibrated radius: 0.05211427

fig, axes = plt.subplots()
axes.loglog(optimiser.progress["count"], optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig("point_discharge2d_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.loglog(optimiser.progress["count"][:-1], optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig("point_discharge2d_qoi.jpg", bbox_inches="tight")

# .. figure:: point_discharge2d_control.jpg
#    :figwidth: 80%
#    :align: center
#
# .. figure:: point_discharge2d_qoi.jpg
#    :figwidth: 80%
#    :align: center
#
#    We get convergence in a small number of iterations thanks to the adaptive step size
#    selection.
#
#    Finally, solve the forward problem to visualise the calibrated radius, which looks
#    much closer to the analytical solution. ::

solutions = mesh_seq.solve_forward()
print(f"Calibrated radius: {float(mesh_seq.controls["r"].tape_value()):.8f}")

plot_kwargs = {
    "levels": np.linspace(0, 1.65, 33),
    "figsize": (10, 3),
    "cmap": "coolwarm",
}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Calibrated forward solution")
fig.savefig("point_discharge2d-forward_calibrated.jpg")

# .. figure:: point_discharge2d-forward_calibrated.jpg
#    :figwidth: 80%
#    :align: center
#
# This tutorial can be dowloaded as a `Python script <point_discharge2d.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography::
#    :filter: docname in docnames
