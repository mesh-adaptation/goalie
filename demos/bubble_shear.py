# Goal-oriented mesh adaptation for a 2D time-dependent problem
# =============================================================

# In a `previous demo <./bubble_shear.py.html>`__, we considered the advection of a
# scalar concentration field in a non-uniform background velocity field. The demo
# concluded with labelling the problem as a good candidate for a goal-oriented mesh
# adaptation approach. This is what we set out to do in this demo.
#
# We will use the same setup as in the original demo, but a small modifiction to the
# :meth:`get_form` function is necessary in order to take into account a prescribed
# time-dependent background velocity field which is not passed to
# :class:`GoalOrientedMeshSeq`. ::

from animate.adapt import adapt
from animate.metric import RiemannianMetric
from firedrake import *

from goalie_adjoint import *

T = 6.0


def velocity_expression(mesh, t):
    x, y = SpatialCoordinate(mesh)
    u_expr = as_vector(
        [
            2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / T),
            -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / T),
        ]
    )
    return u_expr


fields = ["c"]


def get_function_spaces(mesh):
    return {"c": FunctionSpace(mesh, "CG", 1)}


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


def get_solver(mesh_seq):
    def solver(index):
        Q = mesh_seq.function_spaces["c"][index]
        V = VectorFunctionSpace(mesh_seq[index], "CG", 1)
        R = FunctionSpace(mesh_seq[index], "R", 0)

        c, c_ = mesh_seq.fields["c"]

        tp = mesh_seq.time_partition
        t_start, t_end = tp.subintervals[index]
        t = Function(R).assign(t_start)
        dt = Function(R).assign(tp.timesteps[index])
        theta = Function(R).assign(0.5)  # Crank-Nicolson implicitness

        # Compute the velocity field at t_start
        u = Function(V)
        u_ = Function(V)
        u_expression = velocity_expression(mesh_seq[index], t)
        u_.interpolate(u_expression)

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
        mesh_seq.read_forms({"c": F})

        bcs = DirichletBC(mesh_seq.function_spaces["c"][index], 0.0, "on_boundary")
        nlvp = NonlinearVariationalProblem(F, c, bcs=bcs)
        nlvs = NonlinearVariationalSolver(nlvp, ad_block_tag="c")

        # Time integrate from t_start to t_end
        while float(t) < t_end + 0.5 * float(dt):
            # Update the background velocity field at the current timestep
            u.interpolate(u_expression)

            # Solve the advection equation
            nlvs.solve()
            yield

            # Update the 'lagged' concentration and velocity field
            c_.assign(c)
            u_.assign(u)
            t.assign(t + dt)

        return {"c": c}

    return solver


# In the `first demo <bubble_shear.py>`__ where we considered this problem, the
# :meth:`get_form` method was only called from within the :meth:`get_solver`, so we were
# able to pass the velocity field as an argument when calling :meth:`get_form`. However,
# in the context of goal-oriented mesh adaptation,  the :meth:`get_form` method is
# called from within :class:`GoalOrientedMeshSeq` while computing error indicators.
# There, only those fields that are passed to :class:`GoalOrientedMeshSeq` are passed to
# :meth:`get_form`. Currently, Goalie does not consider passing non-prognostic fields
# to this method during the error-estimation step, so the velocity would not be updated
# in time. To account for this, we need to add some code for updating the velocity
# field when it is not present in the dictionary of fields passed. ::


# To compute the adjoint, we must also define the goal functional. As motivated in
# the previous demo, we will use the L2 norm of the difference between the
# concentration field at :math:`t=0` and :math:`t=T/2`, i.e. the simulation end time.
#
# .. math::
#   J(c) := \int_{\Omega} (c(x, y, T/2) - c_0(x, y))^2 \, dx \, dy. ::


def get_qoi(mesh_seq, index):
    def qoi():
        c0 = mesh_seq.get_initial_condition()["c"]
        c0_proj = project(c0, mesh_seq.function_spaces["c"][index])
        c = mesh_seq.fields["c"][0]

        J = (c - c0_proj) * (c - c0_proj) * dx
        return J

    return qoi


# We must also define the adaptor function to drive the mesh adaptation process. Here
# we compute the anisotropic DWR metric which requires us to compute the hessian of the
# tracer concentration field. We combine each metric in a subinterval by intersection.
# We will only run two iterations of the fixed point iteration algorithm so we do not
# ramp up the complexity. ::


def adaptor(mesh_seq, solutions, indicators):
    iteration = mesh_seq.fp_iteration
    mp = {
        "dm_plex_metric": {
            "target_complexity": 1000,
            "p": 1.0,
            "h_min": 1e-04,
            "h_max": 1.0,
        }
    }

    metrics = []
    for i, mesh in enumerate(mesh_seq):
        c_solutions = solutions["c"]["forward"][i]

        P1_ten = TensorFunctionSpace(mesh, "CG", 1)
        subinterval_metric = RiemannianMetric(P1_ten)

        for j, c_sol in enumerate(c_solutions):
            # Recover the Hessian of the forward solution
            hessian = RiemannianMetric(P1_ten)
            hessian.compute_hessian(c_sol)

            metric = RiemannianMetric(P1_ten)
            metric.set_parameters(mp)

            # Compute an anisotropic metric from the error indicator field and Hessian
            metric.compute_anisotropic_dwr_metric(indicators["c"][i][j], hessian)

            # Combine the metrics from each exported timestep in the subinterval
            subinterval_metric.intersect(metric)
        metrics.append(metric)

    # Normalise the metrics in space and time
    space_time_normalise(metrics, mesh_seq.time_partition, mp)

    # Apply mesh adaptation
    # mesh_seq.set_meshes(map(adapt, mesh_seq, metrics))
    for i, metric in enumerate(metrics):
        mesh_seq[i] = adapt(mesh_seq[i], metric)
    num_dofs = mesh_seq.count_vertices()
    num_elem = mesh_seq.count_elements()
    pyrint(f"Fixed point iteration {iteration}:")
    pyrint(f"  qoi: {mesh_seq.qoi_values[-1]:.2e}")
    for i, (ndofs, nelem, metric) in enumerate(zip(num_dofs, num_elem, metrics)):
        pyrint(
            f"  subinterval {i}, complexity: {metric.complexity():4.0f}"
            f", dofs: {ndofs:4d}, elements: {nelem:4d}"
        )

    return True


# Finally, we can define the :class:`GoalOrientedMeshSeq` and run the fixed point
# iteration. For the purposes of this demo, we divide the time interval into 6
# subintervals and only run two iterations of the fixed point iteration, which is not
# enough to reach convergence. ::

n = 50
dt = 0.01
maxiter = 2  # maximum number of fixed point iterations

num_subintervals = 6
meshes = [UnitSquareMesh(n, n) for _ in range(num_subintervals)]
end_time = T / 2
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    fields,
    num_timesteps_per_export=5,
)

msq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)
parameters = GoalOrientedAdaptParameters({"maxiter": maxiter})
solutions, indicators = msq.fixed_point_iteration(adaptor, parameters=parameters)

# Let us plot the intermediate and final solutions, as well as the final adapted meshes. ::

import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot

fig, axes = plt.subplots(2, 3, figsize=(7.5, 5), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.tick_params(axis='both', which='major', labelsize=6)

    # Plot the solution at the final subinterval timestep
    time = time_partition.subintervals[i][-1]
    tripcolor(solutions["c"]["forward"][i][-1], axes=ax, cmap="coolwarm")
    triplot(msq[i], axes=ax, interior_kw={"linewidth": 0.08})
    ax.annotate(f"t={time:.2f}", (0.05, 0.05), color="white")

fig.tight_layout(pad=0.3)
fig.savefig("bubble_shear-goal_oriented.jpg", dpi=300, bbox_inches="tight")

fig, axes, tcs = plot_indicator_snapshots(
    indicators,
    time_partition,
    "c",
)
fig.savefig("bubble_shear-goal_oriented-indicators.jpg", dpi=300, bbox_inches="tight")

# .. figure:: bubble_shear-goal_oriented.jpg
#    :figwidth: 80%
#    :align: center
#
# Compared to the solution obtained on a fixed uniform mesh in the previous demo, we
# observe that the final solution at :math:`t=T/2` better matches the initial tracer
# bubble :math:`c_0` and is less diffused. Despite employing only two fixed
# point iterations, the goal-oriented mesh adaptation process was still able to
# significantly improve the accuracy of the solution while reducing the number of
# degrees of freedom by half.
#
# As shown in the above figure, the bubble experiences extreme deformation and requires
# frequent adaptation to be resolved accurately (surely more than 5, as in this demo!).
# We encourage users to experiment with different numbers of subintervals, number of
# exports per subinterval, adaptor functions, and other parameters to explore the
# convergence of the fixed point iteration. Another interesting experiment would be to
# compare the impact of switching to an explicit time integration and using a smaller
# timestep to maintain numerical stability (look at CFL condition).
#
# This tutorial can be dowloaded as a `Python script <bubble_shear-goal_oriented.py>`__.
