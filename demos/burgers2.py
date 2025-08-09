# Adjoint Burgers equation on two meshes
# =========================================
#
# This demo solves the same adjoint problem as `the previous one
# <./burgers1.py.html>`__, but now using two subintervals. There
# is still no error estimation or mesh adaptation; the same mesh
# is used in each case to verify that the framework works.

from firedrake import *

from goalie import *

set_log_level(DEBUG)

# Redefine the meshes and  field metadata as in previous demos, as well as all the
# getter functions. In this case, we make the default `diagonal="left"` keyword argument
# to :class:`~.UnitSquareMesh` explicit. (See later.) ::

n = 32
mesh = UnitSquareMesh(n, n, diagonal="left")
fields = [Field("u", family="Lagrange", degree=2, vector=True)]


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
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="u")
            yield

            u_.assign(u)
            t += dt

    @annotate_qoi
    def get_qoi(self, i):
        def end_time_qoi():
            u = self.field_functions["u"][0]
            return inner(u, u) * ds(2)

        return end_time_qoi


# This time, the ``TimePartition`` is defined on **two** subintervals. ::

end_time = 0.5
dt = 1 / n
num_subintervals = 2
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    num_timesteps_per_export=2,
)
mesh_seq = MeshSeq([mesh for _ in range(num_subintervals)])
solver = BurgersSolver(time_partition, mesh_seq, qoi_type="end_time")
solutions = solver.solve_adjoint()

# Recall that :func:`solve_forward` runs the solver on each subinterval and
# uses conservative projection to transfer inbetween. This also happens in
# the forward pass of :func:`solve_adjoint`, but is followed by running the
# *adjoint* of the solver on each subinterval *in reverse*. The adjoint of
# the conservative projection operator is applied to transfer adjoint solution
# data between meshes in this case. If you think about the matrix
# representation of a projection operator then this effectively means taking
# the transpose. Again, the meshes (and hence function spaces) are identical,
# so the transfer is just the identity.
#
# Snapshots of the adjoint solution are again plotted using the
# :func:`plot_snapshots` utility function. ::

fig, axes, tcs = plot_snapshots(
    solutions, time_partition, "u", "adjoint", levels=np.linspace(0, 0.8, 9)
)
fig.savefig("burgers2-end_time.jpg")

# .. figure:: burgers2-end_time.jpg
#    :figwidth: 90%
#    :align: center
#
# The adjoint solution fields at each time level appear to match
# those due to the previous demo at each timestep. That they actually
# do coincide is checked in Goalie's test suite.
#
# .. rubric:: Exercise
#
# Note that the keyword argument ``diagonal="left"`` was passed to the
# ``UnitSquareMesh`` constructor in this example, defining which way
# the diagonal lines in the uniform mesh should go. Instead of having
# both function spaces defined on this mesh, try defining the second
# one in a :math:`\mathbb P2` space defined on a **different** mesh
# which is constructed with ``diagonal="right"``. How does the adjoint
# solution change when the solution is trasferred between different
# meshes? In this case, the mesh-to-mesh transfer operations will no
# longer simply be identities.
#
# In the `next demo <./burgers_time_integrated.py.html>`__, we solve
# the same problem but with a QoI involving an integral in time, as
# well as space.
#
# This demo can also be accessed as a `Python script <burgers2.py>`__.
