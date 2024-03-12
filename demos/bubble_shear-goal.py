# Goal-oriented mesh adaptation for a 2D time-dependent problem
# ===================================================

# In a `previous demo <./bubble_shear.py.html>`__, we considered the advection of a scalar
# concentration field in a non-uniform background velocity field. The demo concluded with
# labelling the problem as a good candidate for a goal-oriented mesh adaptation approach.
# This is what we set out to do in this demo.
#
# We will use the same setup as in the previous demo, but a small modifiction to the
# :meth:`get_form` function is necessary in order to take into account a prescribed
# time-dependent background velocity field which is not passed to :class:`GoalOrientedMeshSeq`. ::

from firedrake import *
from goalie_adjoint import *
from animate.metric import RiemannianMetric
from animate.adapt import adapt

period = 6.0


def velocity_expression(x, y, t):
    u_expr = as_vector(
        [
            2 * sin(pi * x) ** 2 * sin(2 * pi * y) * cos(2 * pi * t / period),
            -sin(2 * pi * x) * sin(pi * y) ** 2 * cos(2 * pi * t / period),
        ]
    )
    return u_expr


fields = ["c"]


def get_function_spaces(mesh):
    return {"c": FunctionSpace(mesh, "CG", 1)}


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
        t = t_start
        while t < t_end - 0.5 * dt:
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


# In the previous demo, the :meth:`get_form` method was only called from within the
# :meth:`get_solver`, so we were able to pass the velocity field as an argument when
# calling :meth:`get_form`. However, in the context of goal-oriented mesh adaptation,
# the :meth:`get_form` method is called from within :class:`GoalOrientedMeshSeq`
# while computing error indicators. There, only those fields that are passed to
# :class:`GoalOrientedMeshSeq` are passed to :meth:`get_form`. This means that the velocity
# field would not be updated in time. In such a case, we will manually compute the
# velocity field using the current simulation time, which will be passed automatically
# as an :arg:`err_ind_time` kwarg in :meth:`GoalOrientedMeshSeq.indicate_errors`. ::


def get_form(mesh_seq):
    def form(index, form_fields, err_ind_time=None):
        Q = mesh_seq.function_spaces["c"][index]
        c, c_ = form_fields["c"]

        if "u" in form_fields:
            u, u_ = form_fields["u"]
        else:
            x, y = SpatialCoordinate(mesh_seq[index])
            V = VectorFunctionSpace(mesh_seq[index], "CG", 1)
            u = Function(V).interpolate(velocity_expression(x, y, err_ind_time))
            u_ = Function(V).interpolate(velocity_expression(x, y, err_ind_time))

        # the rest remains unchanged

        R = FunctionSpace(mesh_seq[index], "R", 0)
        dt = Function(R).assign(mesh_seq.time_partition.timesteps[index])
        theta = Function(R).assign(0.5)  # Crank-Nicolson theta

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


# To compute the adjoint, we must also define the goal functional. As motivated in
# the previous demo, we will use the L2 norm of the difference between the
# concentration field at :math:`t=0` and :math:`t=T/2`, i.e. the simulation end time.
#
# .. math::
#   J(c) := \int_{\Omega} (c(x, y, T/2) - c_0(x, y))^2 \, dx \, dy. ::


def get_qoi(self, sols, index):
    def qoi():
        c0 = self.get_initial_condition()["c"]
        c0_proj = project(c0, self.function_spaces["c"][index])
        c = sols["c"]

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

    complexities = np.zeros(len(mesh_seq))
    for i, metric in enumerate(metrics):
        complexities[i] = metric.complexity()
        mesh_seq[i] = adapt(mesh_seq[i], metric)

    num_dofs = mesh_seq.count_vertices()
    num_elem = mesh_seq.count_elements()
    pyrint(f"Fixed point iteration {iteration}:")
    for i, (complexity, ndofs, nelem) in enumerate(
        zip(complexities, num_dofs, num_elem)
    ):
        pyrint(
            f"  subinterval {i}, complexity: {complexity:4.0f}"
            f", dofs: {ndofs:4d}, elements: {nelem:4d}"
        )

    return True


# Finally, we can define the :class:`GoalOrientedMeshSeq` and run the fixed point iteration.
# For the purposes of this demo, we divide the time interval into 25 subintervals and only
# run two iterations of the fixed point iteration, which is not enough to reach convergence. ::

num_subintervals = 25
n = 50
meshes = [UnitSquareMesh(n, n) for _ in range(num_subintervals)]
end_time = period / 2
dt = 0.01
time_partition = TimePartition(
    end_time,
    len(meshes),
    dt,
    fields,
    num_timesteps_per_export=6,
)

parameters = GoalOrientedMetricParameters({"maxiter": 2})
msq = GoalOrientedMeshSeq(
    time_partition,
    meshes,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_bcs=get_bcs,
    get_form=get_form,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
    parameters=parameters,
)
solutions, indicators = msq.fixed_point_iteration(adaptor)

# Let us plot the intermediate and final solutions, as well as the final adapted meshes.

import matplotlib.pyplot as plt
from firedrake.pyplot import tripcolor, triplot

fig, axes = plt.subplots(3, 3, figsize=(8, 8), sharex=True, sharey=True)
for i, ax in enumerate(axes.flat):
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    tripcolor(solutions["c"]["forward"][int(i * 3)][-1], axes=ax, cmap="coolwarm")
    triplot(msq[int(i * 3)], axes=ax, interior_kw={"linewidth": 0.08})
    time = time_partition.subintervals[int(i * 3)][-1]
    ax.annotate(f"t={time:.2f}", (0.05, 0.05), color="white")
fig.tight_layout(pad=0.4)
fig.savefig("bubble_shear-goal_final_meshes.jpg", dpi=300, bbox_inches="tight")

# .. figure:: bubble_shear-goal_final_meshes.jpg
#    :figwidth: 80%
#    :align: center
#
# Compared to the solution obtained on a fixed uniform mesh in the previous demo, we
# observe that the final solution at :math:`t=T/2` better matches the initial tracer
# bubble :math:`c_0` and is less diffused. Despite employing only two fixed
# point iterations, the goal-oriented mesh adaptation process was still able to
# significantly improve the accuracy of the solution while reducing the number of
# degrees of freedom by half. We encourage further experimentation with the number of
# subintervals, adaptor functions and fixed point iterations necessary to reach convergence.
#
# This tutorial can be dowloaded as a `Python script <bubble_shear-goal.py>`__.
