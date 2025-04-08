# Steady-state PDE-constrained optimisation
# =========================================

# TODO: Text
#
# Recall the steady-state advection-diffusion test case from :cite:`Riadh:2014`. In
# this test case, we solve
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
# TODO: text on optimisation problem
#
# As always, start by importing Firedrake and Goalie. We also import some plotting
# utilities. ::

import matplotlib.pyplot as plt
from firedrake import *
from firedrake.pyplot import *

from goalie_adjoint import *

# We solve the advection-diffusion problem in :math:`\mathbb P1` space. ::
# TODO: text

mesh = RectangleMesh(200, 40, 50, 10)
fields = [
    Field("c", mesh=mesh, family="Lagrange", degree=1, unsteady=False),
    Field("u_x", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("u_y", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("D", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
    Field("r", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False),
]

# For steady-state problems, we do not need to specify :func:`get_initial_condition`
# if the equation is linear. If the equation is nonlinear then this would provide
# an initial guess. By default, all components are initialised to zero.
# TODO: text: starting from 0.1, more general approach to params


def get_initial_condition(mesh_seq):
    P1 = mesh_seq.function_spaces["c"][0]
    R = mesh_seq.function_spaces["r"][0]
    return {
        "c": Function(P1).assign(0.0),
        "u_x": Function(R).assign(1.0),
        "u_y": Function(R).assign(0.0),
        "D": Function(R).assign(0.1),
        "r": Function(R).assign(0.1),  # To be calibrated
    }


# TODO: text
# Point sources are difficult to represent in numerical models. Here we
# follow :cite:`Wallwork:2022` in using a Gaussian approximation. Let
# :math:`(x_0,y_0)=(2,5)` denote the point source location and
# :math:`r=0.05606388` be a radius parameter, which has been calibrated
# so that the finite element approximation is as close as possible to the
# analytical solution, in some sense (see :cite:`Wallwork:2022` for details). ::


# TODO: text
# On its own, a :math:`\mathbb P1` discretisation is unstable for this
# problem. Therefore, we include additional `streamline upwind Petrov
# Galerkin (SUPG)` stabilisation by modifying the test function
# :math:`\psi` according to
#
# .. math::
#    \psi \mapsto \psi + \tau\mathbf u\cdot\nabla\psi,
#
# with stabilisation parameter
#
# .. math::
#    \tau = \min\left(\frac{h}{2\|\mathbf u\|},\frac{h\|\mathbf u\|}{6D}\right),
#
# where :math:`h` measures cell size.
#
# Note that :attr:`mesh_seq.field_functions` now returns a single
# :class:`~firedrake.function.Function` object since the problem is steady, so there is
# no notion of a lagged solution, unlike in previous (time-dependent) demos.
# With these ingredients, we can now define the :meth:`get_solver` method. Don't forget
# to apply the corresponding `ad_block_tag` to the solve call. Additionally, we must
# communicate the defined variational form to ``mesh_seq`` using the
# :meth:`mesh_seq.read_form()` method for Goalie to utilise it during error indication.
# ::


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


# As in the motivation for the manual, we consider a quantity of interest that
# integrates the tracer concentration over a circular "receiver" region. Since
# there is no time dependence, the QoI looks just like an ``"end_time"`` type QoI. ::
# TODO: text


def analytical_solution(mesh_seq, index):
    """
    Analytical solution for point-discharge with diffusion problem.
    """
    # TODO: explain radius param, citation
    u_x = mesh_seq.field_functions["u_x"]
    D = mesh_seq.field_functions["D"]
    r = mesh_seq.field_functions["r"]
    if float(r) < 0.0:
        raise ValueError("QoI radius parameter must be positive.")
    x, y = SpatialCoordinate(mesh_seq[index])
    x0, y0 = 2, 5
    Pe = 0.5 * u_x / D  # Mesh Peclet number
    q = 1.0  # TODO: Explain
    r_thresh = max_value(sqrt((x - x0) ** 2 + (y - y0) ** 2), r)
    return 0.5 * q / (pi * D) * exp(Pe * (x - x0)) * bessk0(Pe * r_thresh)


def get_qoi(mesh_seq, index):
    def qoi():
        c = mesh_seq.field_functions["c"]
        c_ana = analytical_solution(mesh_seq, index)
        return (c - c_ana) ** 2 * dx

    return qoi


# Finally, we can set up the problem. Instead of using a :class:`TimePartition`,
# we use the subclass :class:`TimeInstant`, whose only input is the field list. ::
# TODO: text

time_partition = TimeInstant(fields)

# When creating the :class:`MeshSeq`, we need to set the ``"qoi_type"`` to
# ``"steady"``. ::
# TODO: text

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

plot_kwargs = {"levels": 50, "figsize": (10, 3), "cmap": "coolwarm"}
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Forward solution")
fig.savefig("point_discharge2d-forward_uncalibrated.jpg")
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "adjoint", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Adjoint solution")
fig.savefig("point_discharge2d-adjoint.jpg")

J = mesh_seq.J
print(f"r = {float(mesh_seq.field_functions["r"]):.4e}")
print(f"J = {J:.4e}")
print(f"dJ/dr = {float(mesh_seq.gradient["r"]):.4e}")

# Plot the analytical solution
# TODO: temp hack
solutions["c"]["forward"][0][0].interpolate(analytical_solution(mesh_seq, 0))
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Analytical solution")
fig.savefig("point_discharge2d-analytical.jpg")

# Note that the gradient with respect to the intitial condition doesn't make sense in
# this example because we have a steady-state, linear PDE, wherein the solution doesn't
# depend on the initial condition. ::
#
# As before, the forward solution is driven by a point source, which is advected from
# left to right and diffused uniformly in all directions.
#
# .. figure:: point_discharge2d-forward.jpg
#    :figwidth: 80%
#    :align: center
#
# TODO: text
# The adjoint solution, on the other hand, is driven by a source term at the
# `receiver` and is advected from right to left. It is also diffused uniformly
# in all directions.
#
# .. figure:: point_discharge2d-adjoint.jpg
#    :figwidth: 80%
#    :align: center
#
# TODO: text

parameters = OptimisationParameters({"lr": 0.015, "maxiter": 100})
optimiser = QoIOptimiser(mesh_seq, "r", parameters, method="gradient_descent")
optimiser.minimise()

fig, axes = plt.subplots()
axes.plot(optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig("point_discharge2d_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig("point_discharge2d_qoi.jpg", bbox_inches="tight")

solutions = mesh_seq.solve_adjoint()
J = mesh_seq.J
print(f"Optimised QoI: {J:.4e}")
print(f"Optimised control: {float(mesh_seq.controls["r"].tape_value()):.4f}")
fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "c", "forward", **plot_kwargs
)
fig.colorbar(tcs[0][0], orientation="horizontal", pad=0.2)
axes.set_title("Forward solution")
fig.savefig("point_discharge2d-forward_calibrated.jpg")
# TODO: use same colorbar scale and ticks as analytical solution

# This tutorial can be dowloaded as a `Python script <point_discharge2d.py>`__.
#
#
# .. rubric:: References
#
# .. bibliography::
#    :filter: docname in docnames
