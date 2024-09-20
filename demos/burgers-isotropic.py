# Burgers equation with Hessian-based mesh adaptation
# ===================================================

# Yet again, we revisit the Burgers equation example. This time, we apply a goal-oriented
# isotropic mesh adaptation. The interesting thing about this demo is that, unlike the
# `previous demo <./point_discharge2d-goal_oriented.py.html>`__ and its predecessor,
# we consider the time-dependent case. Moreover, we consider a :class:`MeshSeq` with
# multiple subintervals and hence multiple meshes to adapt.
#
# As before, we copy over what is now effectively boiler plate to set up our problem. ::

from firedrake import *
from animate.adapt import adapt
from animate.metric import RiemannianMetric
from goalie import *
from goalie_adjoint import *
import matplotlib.pyplot as plt

from firedrake.__future__ import interpolate


fields = ["u"]


def get_function_spaces(mesh):
    return {"u": VectorFunctionSpace(mesh, "CG", 2)}


def get_form(mesh_seq):
    def form(index, solutions):
        u, u_ = solutions["u"]
        P = mesh_seq.time_partition

        # Define constants
        R = FunctionSpace(mesh_seq[index], "R", 0)
        dt = Function(R).assign(P.timesteps[index])
        nu = Function(R).assign(0.0001)

        # Setup variational problem
        v = TestFunction(u.function_space())
        F = (
            inner((u - u_) / dt, v) * dx
            + inner(dot(u, nabla_grad(u)), v) * dx
            + nu * inner(grad(u), grad(v)) * dx
        )
        return {"u": F}

    return form


def get_solver(mesh_seq):
    def solver(index, ic):
        function_space = mesh_seq.function_spaces["u"][index]
        u = Function(function_space, name="u")
        solution_map = {"u": u}

        # Initialise 'lagged' solution
        u_ = Function(function_space, name="u_old")
        u_.assign(ic["u"])

        # Define form
        F = mesh_seq.form(index, {"u": (u, u_)})["u"]

        # Time integrate from t_start to t_end
        t_start, t_end = mesh_seq.subintervals[index]
        dt = mesh_seq.time_partition.timesteps[index]
        t = t_start
        qoi = mesh_seq.get_qoi(solution_map, index)
        while t < t_end - 1.0e-05:
            solve(F == 0, u, ad_block_tag="u")
            mesh_seq.J += qoi(t)
            u_.assign(u)
            t += dt
        return solution_map

    return solver


def get_initial_condition(mesh_seq):
    fs = mesh_seq.function_spaces["u"][0]
    x, y = SpatialCoordinate(mesh_seq[0])
    return {"u": assemble(interpolate(as_vector([sin(pi * x), 0]), fs))}


def get_qoi(mesh_seq, solutions, i):
    R = FunctionSpace(mesh_seq[i], "R", 0)
    dt = Function(R).assign(mesh_seq.time_partition[i].timestep)

    def time_integrated_qoi(t):
        u = solutions["u"]
        return dt * inner(u, u) * ds(2)

    return time_integrated_qoi


n = 32
meshes = [UnitSquareMesh(n, n, diagonal="left"), UnitSquareMesh(n, n, diagonal="left")]
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

params = GoalOrientedMetricParameters(
    {
        "element_rtol": 0.001,
        "maxiter": 35 if os.environ.get("GOALIE_REGRESSION_TEST") is None else 3,
    }
)

mesh_seq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="time_integrated",
    parameters=params,
)

# As in the previous adaptation demos, the most important part is the adaptor function.
# The one used here takes a similar form, except that we need to handle multiple meshes
# and metrics.
# We then time integrate these to give the metric contribution from each subinterval.
# Given that we use a simple implicit Euler method for time integration
# in the PDE, we do the same here, too.
#
# Goalie provides functionality to normalise a list of metrics using *space-time*
# normalisation. This ensures that the target complexity is attained on average across
# all timesteps.
#
# Note that when adapting the mesh, we need to be careful to check whether convergence
# has already been reached on any of the subintervals. If so, the adaptation step is
# skipped. ::


def adaptor(mesh_seq, solutions, indicators):
    metrics = []
    complexities = []

    # Ramp the target average metric complexity per timestep
    base, target, iteration = 400, 1000, mesh_seq.fp_iteration
    mp = {
        "dm_plex_metric": {
            "target_complexity": ramp_complexity(base, target, iteration),
            "p": 1.0,
            "h_min": 1.0e-04,
            "h_max": 1.0,
        }
    }

    # Construct the metric on each subinterval
    for i, mesh in enumerate(mesh_seq):
        sols = solutions["u"]["forward"][i]
        dt = mesh_seq.time_partition.timesteps[i]

        # Define the Riemannian metric
        P1_ten = TensorFunctionSpace(mesh, "CG", 1)

        # At each timestep, recover metric of the solution
        # vector. Then time integrate over the contributions
        _metrics = []

        for j, sol in enumerate(sols):

            # get local indicator
            indi = indicators["u"][i][j]
            # local instance of Riemanian metric
            _metric = RiemannianMetric(P1_ten)

            # reset parameters
            _metric.set_parameters(mp)

            # Deduce an isotropic metric from the error indicator field
            _metric.compute_isotropic_metric(error_indicator=indi, interpolant="L2")
            _metric.normalise()

            # append the metric for the step in the time partition
            _metrics.append(_metric)

        # set the first metric to the base
        metric = _metrics[0]
        # all the other metrics
        metric.average(*_metrics[1:], weights=[dt] * len(_metrics))

        metrics.append(metric)

    # Apply space time normalisation
    space_time_normalise(metrics, mesh_seq.time_partition, mp)

    # Adapt each mesh w.r.t. the corresponding metric, provided it hasn't converged
    for i, metric in enumerate(metrics):
        if not mesh_seq.converged[i]:
            mesh_seq[i] = adapt(mesh_seq[i], metric)
        complexities.append(metric.complexity())
    num_dofs = mesh_seq.count_vertices()
    num_elem = mesh_seq.count_elements()
    pyrint(f"fixed point iteration {iteration + 1}:")
    for i, (complexity, ndofs, nelem) in enumerate(
        zip(complexities, num_dofs, num_elem)
    ):
        pyrint(
            f"  subinterval {i}, complexity: {complexity:4.0f}"
            f", dofs: {ndofs:4d}, elements: {nelem:4d}"
        )

    # Plot each intermediate adapted mesh
    fig, axes = mesh_seq.plot()
    for i, ax in enumerate(axes):
        ax.set_title(f"Subinterval {i + 1}")
    fig.savefig(f"burgers-hessian_mesh{iteration + 1}.jpg")
    plt.close()

    # Since we have two subintervals, we should check if the target complexity has been
    # (approximately) reached on each subinterval
    continue_unconditionally = np.array(complexities) < 0.90 * target
    return continue_unconditionally


# With the adaptor function defined, we can call the fixed point iteration method. ::

solutions = mesh_seq.fixed_point_iteration(
    adaptor,
    enrichment_kwargs={
        "enrichment_method": "h",
    },
)

# Here the output should look something like
#
# .. code-block:: console
#
# fixed point iteration 1:
#   subinterval 0, complexity:  449, dofs:  619, elements: 1162
#   subinterval 1, complexity:  350, dofs:  451, elements:  833
# fixed point iteration 2:
#   subinterval 0, complexity:  654, dofs:  778, elements: 1453
#   subinterval 1, complexity:  539, dofs:  643, elements: 1199
# fixed point iteration 3:
#   subinterval 0, complexity:  884, dofs:  945, elements: 1789
#   subinterval 1, complexity:  709, dofs:  795, elements: 1487
# QoI converged after 4 iterations under relative tolerance 0.001.

# Finally, let's plot the adapted meshes. ::

fig, axes = mesh_seq.plot()
for i, ax in enumerate(axes):
    ax.set_title(f"Subinterval {i + 1}")
fig.savefig("burgers-go_iso_mesh.jpg")
plt.close()

# .. figure:: burgers-iso_mesh.jpg
#    :figwidth: 100%
#    :align: center
#
# Recall that the Burgers problem is quasi-1D, since the analytical solution varies in
# the :math:`x`-direction, but is constant in the :math:`y`-direction. As such, we can
# affort to have lower resolution in the :math:`y`-direction in adapted meshes. That
# this occurs is clear from the plots above. The solution moves to the right, becoming
# more densely distributed near to the right-hand boundary. This can be seen by
# comparing the second mesh against the first.
#
# .. rubric:: Exercise
#
#
# This demo can also be accessed as a `Python script <burgers-isotropic.py>`__.
