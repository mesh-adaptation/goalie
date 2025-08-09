# Adjoint Burgers equation with a time integrated QoI
# ===================================================
#
# So far, we only considered a quantity of interest corresponding to a spatial integral
# at the end time. For some problems, it is more suitable to have a QoI which integrates
# in time as well as space.
#
# Begin by importing from Firedrake and Goalie.

from firedrake import *

from goalie import *

# Redefine the mesh, fields and ``get_initial_condition`` function as in `the previous
# demo <./burgers2.py.html>`__. ::

n = 32
mesh = UnitSquareMesh(n, n)
fields = [Field("u", family="Lagrange", degree=2, vector=True)]


# The solver needs to be modified slightly in order to take account of time dependent
# QoIs. The Burgers solver uses backward Euler timestepping. The corresponding
# quadrature routine is like the midpoint rule, but takes the value from the next
# timestep, rather than the average between that and the current value. As such, the QoI
# may be computed by simply incrementing the :attr:`J` attribute of the
# :class:`AdjointMeshSeq` as follows. ::


class BurgersSolver(AdjointSolver):
    def get_initial_condition(self):
        fs = self.function_spaces["u"][0]
        x, y = SpatialCoordinate(self.meshes[0])
        return {"u": Function(fs).interpolate(as_vector([sin(pi * x), 0]))}

    def get_solver(self, index):
        # Get the current and lagged solutions
        u, u_ = self.field_functions["u"]

        # Define constants
        R = FunctionSpace(self.meshes[index], "R", 0)
        dt = Function(R).assign(self.time_partition.timesteps[index])
        nu = Function(R).assign(0.0001)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )

        # Time integrate from t_start to t_end
        t_start, t_end = self.time_partition.subintervals[index]
        dt = self.time_partition.timesteps[index]
        t = t_start
        qoi = self.get_qoi(index)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="u")
            yield
            self.J += qoi(t)

            u_.assign(u)
            t += dt

    @annotate_qoi
    def get_qoi(self, i):
        R = FunctionSpace(self.meshes[i], "R", 0)
        dt = Function(R).assign(self.time_partition.timesteps[i])

        def time_integrated_qoi(t):
            u = self.field_functions["u"][0]
            return dt * inner(u, u) * ds(2)

        return time_integrated_qoi


# The QoI is effectively just a time-integrated version of the one previously seen.
#
# .. math::
#    J(u) = \int_0^{T_{\mathrm{end}}} \int_0^1
#    \mathbf u(1,y,t) \cdot \mathbf u(1,y,t)
#    \;\mathrm dy\;\mathrm dt.
#
# Note that in this case we multiply by the timestep. It is wrapped in a
# :class:`Function` from `'R'` space to avoid recompilation if the value is changed. ::


# We use the same time partitioning as in `the previous demo <./burgers2.py.html>`__,
# except that we export every timestep rather than every other timestep. ::

end_time = 0.5
dt = 1 / n
num_subintervals = 2
time_partition = TimePartition(
    end_time, num_subintervals, dt, fields, num_timesteps_per_export=1
)

# The only difference when defining the :class:`AdjointMeshSeq` is that we specify
# ``qoi_type="time_integrated"``, rather than ``qoi_type="end_time"``. ::

mesh_seq = [mesh for _ in range(num_subintervals)]
solver = BurgersSolver(time_partition, mesh_seq, qoi_type="time_integrated")
solutions = solver.solve_adjoint()

fig, axes, tcs = plot_snapshots(solutions, time_partition, "u", "adjoint")
fig.savefig("burgers-time_integrated.jpg")

# .. figure:: burgers-time_integrated.jpg
#    :figwidth: 90%
#    :align: center
#
# With a time-integrated QoI, the adjoint problem has a source term at the right-hand
# boundary, rather than a instantaneous pulse at the terminal time. As such, the adjoint
# solution field accumulates at the right-hand boundary, as well as propagating
# westwards.
#
# In the `next demo <./burgers_oo.py.html>`__, we solve the Burgers problem one last
# time, but using an object-oriented approach.
#
# This demo can also be accessed as a `Python script <burgers_time_integrated.py>`__.
