# Advection of a tracer bubble in a non-uniform shear flow
# ========================================================

# In this demo, we consider the advection of a scalar concentration field, similarly to
# the `solid body rotation demo <./solid_body_rotation.py.html>`__. However, we now
# consider a background velocity field that is not uniform in time. We further prescribe
# homogeneous Dirichlet boundary conditions for the concentration, so in this demo we
# solve the following advection problem:
#
# .. math::
#   \begin{array}{rl}
#       \frac{\partial c}{\partial t} + \nabla\cdot(\mathbf{u}c)=0& \text{in}\:\Omega\\
#       c=0 & \text{on}\:\partial\Omega,\\
#       c=c_0(x,y) & \text{at}\:t=0,
#   \end{array},
#
# where :math:`c=c(x,y,t)` is the sought tracer concentration,
# :math:`\mathbf{u}=\mathbf{u}(x,y,t)` is the background velocity field, and
# :math:`\Omega=[0, 1]^2` is the spatial domain of interest.
#
# First, we import Firedrake and Goalie. ::

from firedrake import *
from goalie import *

# We begin by definining the background velocity field :math:`\mathbf{u}(x, y, t)`.
# Specifically, we choose the velocity field that is periodic in time, such as
# in :cite:<Barral:2016>, and is given by
#
# .. math::
#   \mathbf{u}(x, y, t) := \cos(2\pi t/T)\left(2\sin^2(\pi x)\sin(2\pi y), -\sin(2\pi x)\sin^2(\pi y) \right),
#
# where :math:`T` is the period. At each timestep of the simulation we will have to
# update this field so we define a function that will return this vector expression. ::

period = 6.0


def velocity_expression(x, y, t):
    u_expr = as_vector(
        [
            2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / period),
            -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / period),
        ]
    )
    return u_expr


# However, since the background velocity :math:`\mathbf{u}` is already known, we do not
# solve for it and pass it as a field to :class:`MeshSeq`. Instead, we pass only the
# tracer concentration :math:`c` and, for now, avoid specifying how to handle the
# velocity field. Therefore, at this stage we only define how to build the
# :class:`FunctionSpace` for the tracer concentration. ::

fields = ["c"]


def get_function_spaces(mesh):
    return {"c": FunctionSpace(mesh, "CG", 1)}


# We proceed similarly with prescribing initial and boundary conditions. At :math:`t=0`,
# we initialise the tracer concentration :math:`c_0 = c(x, y, 0)` to be :math:`1` inside
# a circular region of radius :math:`r_0=0.15` centred at :math:`(x_0, y_0)=(0.5, 0.65)`
# and :math:`0` elsewhere in the domain. Note that this is a discontinuous function
# which will not be represented well on a coarse uniform mesh. ::


def get_bcs(mesh_seq):
    def bcs(index):
        return [DirichletBC(mesh_seq.function_spaces["c"][index], 0.0, "on_boundary")]

    return bcs


def ball_initial_condition(x, y):
    ball_r0, ball_x0, ball_y0 = 0.15, 0.5, 0.65
    r = sqrt(pow(x - ball_x0, 2) + pow(y - ball_y0, 2))
    return conditional(r < ball_r0, 1.0, 0.0)


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["c"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    ball = ball_initial_condition(x, y)
    c0 = assemble(interpolate(ball, fs))
    return {"c": c0}


# To compile the weak form, we require both the concentration field :math:`c` and the
# background velocity field :math:`\mathbf{u}` in :meth:`get_form`. In this demo, we
# will always pass both these fields from :meth:`get_solver`. Note that we still do not
# define the velocity field.
# As in the point discharge with diffusion demo, we include additional `streamline
# upwind Petrov Galerkin (SUPG)` stabilisation by modifying the test function
# :math:`\psi`. To advance in time, we use Crank-Nicolson timestepping. ::


def get_form(mesh_seq):
    def form(index, form_fields):
        Q = mesh_seq.function_spaces["c"][index]
        c, c_ = form_fields["c"]
        u, u_ = form_fields["u"]

        R = FunctionSpace(mesh_seq[index], "R", 0)
        dt = Function(R).assign(mesh_seq.time_partition.timesteps[index])
        theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

        # SUPG stabilisation parameter
        D = Function(R).assign(0.1)  # diffusivity coefficient
        h = CellSize(mesh_seq[index])  # mesh cell size
        U = sqrt(dot(u, u))  # velocity magnitude
        tau = 0.5 * h / U
        tau = min_value(tau, U * h / (6 * D))

        phi = TestFunction(Q)
        phi += tau * dot(u, grad(phi))

        a = c * phi * dx + dt * theta * dot(u, grad(c)) * phi * dx
        L = c_ * phi * dx - dt * (1 - theta) * dot(u_, grad(c_)) * phi * dx
        F = a - L

        return {"c": F}

    return form


# Finally, it remains to define the solver. Since :class:`MeshSeq` contains information
# on how to build the function spaces, initial and boundary conditions for the tracer
# concentration, we can simply define them as in the previous demos.
# On the other hand, so far we have not specified how to handle the background velocity
# field. We will do this here. ::


def get_solver(mesh_seq):
    def solver(index, ic):
        tp = mesh_seq.time_partition
        t_start, t_end = tp.subintervals[index]
        dt = tp.timesteps[index]

        # Initialise the concentration fields
        Q = mesh_seq.function_spaces["c"][index]
        c = Function(Q, name="c")
        c_ = Function(Q, name="c_old").assign(ic["c"])

        # Specify the velocity function space
        V = VectorFunctionSpace(mesh_seq[index], "CG", 1)
        u = Function(V)
        u_ = Function(V)

        # Compute the velocity field at t_start and assign it to u_
        x, y = SpatialCoordinate(mesh_seq[index])
        u_.interpolate(velocity_expression(x, y, t_start))

        # We pass both the concentration and velocity Functions to get_form
        form_fields = {"c": (c, c_), "u": (u, u_)}
        F = mesh_seq.form(index, form_fields)["c"]
        nlvp = NonlinearVariationalProblem(F, c, bcs=mesh_seq.bcs(index))
        nlvs = NonlinearVariationalSolver(nlvp, ad_block_tag="c")

        # Time integrate from t_start to t_end
        t = t_start + dt
        while t < t_end + 0.5 * dt:
            # update the background velocity field at the current timestep
            u.interpolate(velocity_expression(x, y, t))

            # solve the advection equation
            nlvs.solve()

            # update the 'lagged' concentration and velocity field
            c_.assign(c)
            u_.assign(u)
            t += dt

        return {"c": c}

    return solver


# Now that we have defined the solver and specified how to build and update the
# background velocity, we are ready to solve the problem. We run the simulation until
# :math:`t=T/2` on a sequence of two coarse meshes. ::

# Reduce the cost of the demo during testing.
test = os.environ.get("GOALIE_REGRESSION_TEST") is not None
n = 50 if not test else 10
dt = 0.01 if not test else 0.025

num_subintervals = 2
meshes = [UnitSquareMesh(n, n) for _ in range(num_subintervals)]
end_time = period / 2
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    num_timesteps_per_export=30,
)

msq = MeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_bcs=get_bcs,
    get_form=get_form,
    get_solver=get_solver,
)
solutions = msq.solve_forward()

# Let us plot the solution :math:`c` at exported timesteps. ::

plot_kwargs = {"cmap": "coolwarm", "levels": 100}
fig, _, _ = plot_snapshots(solutions, time_partition, "c", "forward", **plot_kwargs)
fig.savefig("bubble_shear.jpg")

# .. figure:: bubble_shear.jpg
#    :figwidth: 80%
#    :align: center
#
# In the first part of the simulation, we observe that the initial tracer bubble becomes
# increasingly deformed by the strong shear forces. However, we notice that the
# deformation appears to be reverting. This is due to the periodicity of the velocity
# field :math:`\mathbf{u}`, which we chose to be an odd function of time. This means
# that the velocity field reverses direction at :math:`t=T/2`, and so does the
# deformation of the tracer bubble, which returns to its initial shape.
#
# However, while the tracer bubble at :math:`t=T/2` does resemble a very blurry circular
# shape, it is far from matching the initial condition. This "bluriness" is the result
# of adding the SUPG stabilisation to the weak form, which adds numerical diffusion.
# Adding diffusion is necessary for numerical stability and for preventing oscillations,
# but it also makes the solution irreversible. The amount of difussion added is related
# to the grid Péclet number :math:`Pe = U\,h/2D`: the coarser the mesh is, the more
# diffusion is added. We encourage the reader to verify this by running the simulation
# on a sequence of finer uniform meshes, and to visit the
# `bubble shear mesh adaptation demo <./bubble_shear-goal.py.html>`__, where we use a
# goal-oriented approach to identify areas of the domain that require refinement to
# reduce the numerical diffusion.
#
# This tutorial can be dowloaded as a `Python script <bubble_shear.py>`__.
