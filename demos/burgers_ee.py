# Error estimation for Burgers equation
# =====================================

# So far, we have learnt how to set up :class:`MeshSeq`\s and solve
# forward and adjoint problems. In this demo, we use this functionality
# to perform goal-oriented error estimation.
#
# The fundamental result in goal-oriented error estimation is the
# *dual-weighted residual*,
#
# .. math::
#    J(u)-J(u_h)\approx\rho(u_h,u^*),
#
# where :math:`u` is the solution of a PDE with weak residual
# :math:`\rho(\cdot,\cdot)`, :math:`u_h` is a finite element solution
# and :math:`J` is the quantity of interest (QoI). Here, the *exact*
# solution :math:`u^*` of the associated adjoint problem replaces the test
# function in the second argument of the weak residual. In practice,
# we do not know what this is, of course. As such, it is common practice
# to evaluate the dual weighted residual by approximating the true adjoint
# solution in an enriched finite element space. That is, a superspace,
# obtained by adding more degrees of freedom to the base space. This could
# be done by solving global or local auxiliary PDEs, or by applying patch
# recovery type methods. Currently, only global enrichment is supported in
# Goalie. ::

from firedrake import *

from goalie import *

set_log_level(DEBUG)

# Redefine the meshes, fields and the getter functions as in the first adjoint Burgers
# demo, with two differences:
#
# * We need to specifically define the mesh for each subinterval and pass them as a
#   list. When a single mesh is passed to the :class:`~.MeshSeq` constructor, it is
#   shallow copied, which is insufficient for the :math:`h`-refinement used in the error
#   estimation step. ::
# * We need to call the :meth:`~.GoalOrientedMeshSeq.read_forms()` method in the
#   ``get_solver`` function. This is used to communicate the variational form to the
#   mesh sequence object so that Goalie can utilise it in the error estimation process
#   described above.
#
# ::

n = 32
meshes = [UnitSquareMesh(n, n), UnitSquareMesh(n, n)]
fields = [Field("u", family="Lagrange", degree=2, vector=True)]


def get_solver(mesh_seq):
    def solver(index):
        u, u_ = mesh_seq.field_functions["u"]

        # Define constants
        R = FunctionSpace(mesh_seq[index], "R", 0)
        dt = Function(R).assign(mesh_seq.time_partition.timesteps[index])
        nu = Function(R).assign(0.0001)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )

        # Communicate variational form to mesh_seq
        mesh_seq.read_forms({"u": F})

        # Time integrate from t_start to t_end
        P = mesh_seq.time_partition
        t_start, t_end = P.subintervals[index]
        dt = P.timesteps[index]
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="u")
            yield

            u_.assign(u)
            t += dt

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {"u": Function(fs).interpolate(as_vector([sin(pi * x), 0]))}


def get_qoi(mesh_seq, i):
    def end_time_qoi():
        u = mesh_seq.field_functions["u"][0]
        return inner(u, u) * ds(2)

    return end_time_qoi


# Next, create a :class:`TimePartition`. ::

end_time = 0.5
dt = 1 / n
num_subintervals = len(meshes)
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    num_timesteps_per_export=2,
)

# A key difference between this demo and the previous ones is that we need to
# use a :class:`GoalOrientedMeshSeq` to access the goal-oriented error estimation
# functionality. Note that :class:`GoalOrientedMeshSeq` is a subclass of
# :class:`AdjointMeshSeq`, which is a subclass of :class:`MeshSeq`. ::

mesh_seq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)

# Given the description of the PDE problem in the form of a
# :class:`GoalOrientedMeshSeq`, Goalie is able to extract all of the relevant
# information to automatically compute error estimators. During the computation, we
# solve the forward and adjoint equations over the mesh sequence, as before. In
# addition, we solve the adjoint problem again in an *enriched* finite element space.
# Currently, Goalie supports uniform refinement of the meshes (:math:`h`-refinement) or
# globally increasing the polynomial order (:math:`p`-refinement). Choosing one (or
# both) of these as the ``"enrichment_method"``, we are able to compute error indicator
# fields as follows. ::

solutions, indicators = mesh_seq.indicate_errors(
    enrichment_kwargs={"enrichment_method": "h"}
)

# An error indicator field :math:`i` takes constant values on each mesh element, say
# :math:`i_K` for element :math:`K` of mesh :math:`\mathcal H`. It decomposes
# the global error estimator :math:`\epsilon` into its local contributions.
#
# .. math::
#    \epsilon = \sum_{K\in\mathcal H}i_K \approx \rho(u_h,u^*).
#
# For the purposes of this demo, we plot the solution at each exported
# timestep using the plotting driver function :func:`plot_indicator_snapshots`. ::

fig, axes, tcs = plot_indicator_snapshots(indicators, time_partition, "u", levels=50)
fig.savefig("burgers-ee.jpg")

# .. figure:: burgers-ee.jpg
#    :figwidth: 90%
#    :align: center
#
# We observe that the contributions to the QoI error are estimated to be much higher in
# the right-hand part of the domain than the left. This makes sense, becase the QoI is
# evaluated along the right-hand boundary and we have already seen that the magnitude
# of the adjoint solution tends to be larger in that region, too.
#
# .. rubric:: Exercise
#
# Try running the demo again, but with a ``"time_integrated"`` QoI, rather than an
# ``"end_time"`` one. How do the error indicator fields change in this case?
#
# This demo can also be accessed as a `Python script <burgers_ee.py>`__.
