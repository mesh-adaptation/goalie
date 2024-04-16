# Mantle convection modelling
#############################

# In all demos that we have considered so far, the equations that we have solved all
# involve a time derivative. However, in some cases, we may have a time-dependent
# problem where an equation might not involve a time derivative, but is still
# time-dependent in the sense that it depends on other fields that are time-dependent
# and that have been previously solved for. An example of where this might happen is
# in a continuous pressure projection approach, ice sheet and glaciological modelling,
# mantle convection modelling, etc. In this demo, we illustrate how Goalie can be used
# to solve such problems.

# We consider the problem of a mantle convection in a 2D unit square domain. The
# governing equations and Firedrake implementation are based on the 2D Cartesian
# incompressible isoviscous case from :cite:`Davies:2022`. We refer the reader to the
# paper for a detailed description of the problem and implementation. Here we
# immediately present the governing equations involving a Stokes system and an energy
# equation, which we solve for the velocity :math:`\mathbf{u}`, pressure :math:`p`, and
# temperature :math:`T`:
#
# .. math::
#    \begin{align}
#        \nabla \cdot \mu \left[\nabla \mathbf{u} + (\nabla \mathbf{u})^T \right] - \nabla p + \mathrm{Ra}\,T\,\mathbf{k} &= 0, \\
#        \frac{\partial T}{\partial t} \cdot \mathbf{u}\cdot\nabla T - \nabla \cdot (\kappa\nabla T) &= 0,
#    \end{align}
#
# where :math:`\mu`, :math:`\kappa`, and :math:`\mathrm{Ra}` are constant viscosity,
# thermal diffusivity, and Rayleigh number, respectively, and :math:`\mathbf{k}` is
# the unit vector in the direction opposite to gravity. Note that while the Stokes
# equations do not involve a time derivative, they are still time-dependent as they
# depend on the temperature field :math:`T`, which is time-dependent.
#
# As always, we begin by importing Firedrake and Goalie, and defining constants.

from firedrake import *
from goalie_adjoint import *

Ra, mu, kappa = Constant(1e4), Constant(1.0), Constant(1.0)
k = Constant((0, 1))

# The problem is solved simultaneously for the velocity :math:`\mathbf{u}` and pressure
# :math:`p` using a *mixed* formulation, which was introduced in a `previous demo on
# advection-diffusion reaction <./gray_scott.py.html>`__.

fields = ["up", "T"]


def get_function_spaces(mesh):
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space
    Z = MixedFunctionSpace([V, W])  # Mixed function space for velocity and pressure
    Q = FunctionSpace(mesh, "CG", 1)  # Temperature function space
    return {"up": Z, "T": Q}


# We must prescribe boundary and initial conditions to solve the problem. Note that
# Goalie requires defining the initial condition for the mixed field ``up`` despite
# the equations not involving a time derivative. In this case, the prescribed initial
# condition should be understood as the *initial guess* for the solver.


def get_bcs(mesh_seq):
    def bcs(index):
        Z = mesh_seq.function_spaces["up"][index]
        Q = mesh_seq.function_spaces["T"][index]

        # Boundary IDs
        left, right, bottom, top = 1, 2, 3, 4

        # Boundary conditions for velocity
        bcvx = DirichletBC(Z.sub(0).sub(0), 0, sub_domain=(left, right))
        bcvy = DirichletBC(Z.sub(0).sub(1), 0, sub_domain=(bottom, top))
        # Boundary conditions for temperature
        bctb = DirichletBC(Q, 1.0, sub_domain=bottom)
        bctt = DirichletBC(Q, 0.0, sub_domain=top)

        return {"up": [bcvx, bcvy], "T": [bctb, bctt]}

    return bcs


def get_initial_condition(mesh_seq):
    x, y = SpatialCoordinate(mesh_seq[0])
    T_init = Function(mesh_seq.function_spaces["T"][0])
    T_init.interpolate(1.0 - y + 0.05 * cos(pi * x) * sin(pi * y))
    up_init = Function(mesh_seq.function_spaces["up"][0])  # Zero by default
    return {"up": up_init, "T": T_init}


# The weak forms for the Stokes and energy equations are defined as follows. Note that
# the lagged solution ``up_`` is not used in compiling the form, but is still by
# internal Goalie machinery.


def get_form(mesh_seq):
    def form(index, sols):
        dt = mesh_seq.time_partition.timesteps[index]

        up, up_ = sols["up"]
        u, p = split(up)
        v, w = TestFunctions(mesh_seq.function_spaces["up"][index])

        T, T_ = sols["T"]
        q = TestFunction(mesh_seq.function_spaces["T"][index])

        # Crank-Nicolson time discretisation for temperature
        Ttheta = 0.5 * (T + T_)

        # Stokes equations
        stress = 2 * mu * sym(grad(u))
        F_up = (
            inner(grad(v), stress) * dx
            - div(v) * p * dx
            - (dot(v, k) * Ra * Ttheta) * dx
        )
        F_up += -w * div(u) * dx  # Continuity equation

        # Energy equation
        F_T = (
            q * (T - T_) / dt * dx
            + q * dot(u, grad(Ttheta)) * dx
            + dot(grad(q), kappa * grad(Ttheta)) * dx
        )

        return {"up": F_up, "T": F_T}

    return form


# In the solver, it is important to solve for the fields in the order in which they are
# defined in the ``fields`` list. This is to ensure that the error indicators are
# computed correctly. Therefore, in the time integration loop, we first solve the
# Stokes equations for the velocity and pressure fields, and then solve the energy
# equation for the temperature field.


def get_solver(mesh_seq):
    def solver(index, ic):
        Z = mesh_seq.function_spaces["up"][index]
        up = Function(Z, name="up")
        up_ = Function(Z, name="up_old")

        Q = mesh_seq.function_spaces["T"][index]
        T = Function(Q, name="T")
        T_ = Function(Q, name="T_old")
        T_.assign(ic["T"])

        # Dictionary of weak forms and boundary conditions for both fields
        F = mesh_seq.form(index, {"up": (up, up_), "T": (T, T_)})
        bcs = mesh_seq.bcs(index)

        # Solver parameters dictionary
        solver_parameters = {
            "mat_type": "aij",
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }

        nlvp_up = NonlinearVariationalProblem(F["up"], up, bcs=bcs["up"])
        nlvs_up = NonlinearVariationalSolver(
            nlvp_up,
            solver_parameters=solver_parameters,
            ad_block_tag="up",
        )
        nlvp_T = NonlinearVariationalProblem(F["T"], T, bcs=bcs["T"])
        nlvs_T = NonlinearVariationalSolver(
            nlvp_T,
            solver_parameters=solver_parameters,
            ad_block_tag="T",
        )

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        num_timesteps = P.num_timesteps_per_subinterval[index]
        for _ in range(num_timesteps):
            nlvs_up.solve()
            nlvs_T.solve()
            T_.assign(T)
        return {"up": up, "T": T}

    return solver


# Finally, we define the quantity of interest (QoI) as the square of the velocity field
# :math:`\mathbf{u}`. The QoI will be evaluated at the final time step.


def get_qoi(mesh_seq, sols, i):
    def qoi():
        up = sols["up"]
        u, _ = split(up)
        return dot(u, u) * dx

    return qoi


# We can now create a GoalOrientedMeshSeq object and compute the error indicators.
# For demonstration purposes, we only consider two subintervals and a low number of
# timesteps.

num_subintervals = 2
meshes = [UnitSquareMesh(32, 32, quadrilateral=True) for _ in range(num_subintervals)]

dt = 1e-3
num_timesteps = 40
end_time = dt * num_timesteps
dt_per_export = [10 for _ in range(num_subintervals)]

# To account for the lack of time derivative in the Stokes equations, we use the
# ``field_types`` argument of the ``TimePartition`` object to specify that the ``up``
# field is *steady* (i.e. without a time derivative) and that the ``T`` field is
# *unsteady* (i.e. involves a time derivative). The order in ``field_types`` must
# match the order of the fields in the ``fields`` list (see above).

time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    num_timesteps_per_export=dt_per_export,
    field_types=["steady", "unsteady"],
)

mesh_seq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_bcs=get_bcs,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)

solutions, indicators = mesh_seq.indicate_errors()

# We can plot the error indicator fields for exported timesteps using the in-built
# plotting function ``plot_indicator_snapshots``.

fig, axes, tcs = plot_indicator_snapshots(indicators, time_partition, "T", levels=50)
fig.savefig("mantle_convection.jpg")

# .. figure:: burgers-ee.jpg
#    :figwidth: 90%
#    :align: center
#
# This demo can also be accessed as a `Python script <mantle_convection.py>`__.
