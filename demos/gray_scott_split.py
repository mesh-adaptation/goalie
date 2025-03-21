# Advection-diffusion-reaction with multiple prognostic variables
# ===============================================================
#
# In the the `previous demo <./gray_scott.py.html>`__, we solved the Gray-Scott
# equation using a mixed formulation for the two tracer species. Here, we instead use
# different fields for each of them, treating the corresponding equations separately.
# This considers an additional level of complexity compared with the
# `split solid body rotation demo <./solid_body_rotation_split.py.html>`__ because the
# equations differ in both the diffusion and reaction terms. ::

from firedrake import *

from goalie_adjoint import *

# This time, we have two fields instead of one, as well as two function spaces. ::

field_names = ["a", "b"]
mesh = PeriodicSquareMesh(65, 65, 2.5, quadrilateral=True, direction="both")


def get_function_spaces(mesh):
    return {
        "a": FunctionSpace(mesh, "CG", 1),
        "b": FunctionSpace(mesh, "CG", 1),
    }


# Therefore, the initial condition must be constructed using separate
# :class:`Function`\s. ::


def get_initial_condition(mesh_seq):
    x, y = SpatialCoordinate(mesh_seq[0])
    fs_a = mesh_seq.function_spaces["a"][0]
    fs_b = mesh_seq.function_spaces["b"][0]
    b_init = Function(fs_b).interpolate(
        conditional(
            And(And(1 <= x, x <= 1.5), And(1 <= y, y <= 1.5)),
            0.25 * sin(4 * pi * x) ** 2 * sin(4 * pi * y) ** 2,
            0,
        )
    )
    a_init = Function(fs_a).interpolate(1 - 2 * b_init)
    return {"a": a_init, "b": b_init}


# Correspondingly the solver needs to be constructed from the two parts and must
# include two nonlinear solves at each timestep. ::


def get_solver(mesh_seq):
    def solver(index):
        a, a_ = mesh_seq.fields["a"]
        b, b_ = mesh_seq.fields["b"]

        # Define constants
        R = FunctionSpace(mesh_seq[index], "R", 0)
        dt = Function(R).assign(mesh_seq.time_partition.timesteps[index])
        D_a = Function(R).assign(8.0e-05)
        D_b = Function(R).assign(4.0e-05)
        gamma = Function(R).assign(0.024)
        kappa = Function(R).assign(0.06)

        # Write the two equations in variational form
        psi_a = TestFunction(mesh_seq.function_spaces["a"][index])
        psi_b = TestFunction(mesh_seq.function_spaces["b"][index])
        F_a = (
            psi_a * (a - a_) * dx
            + dt * D_a * inner(grad(psi_a), grad(a)) * dx
            - dt * psi_a * (-a * b**2 + gamma * (1 - a)) * dx
        )
        F_b = (
            psi_b * (b - b_) * dx
            + dt * D_b * inner(grad(psi_b), grad(b)) * dx
            - dt * psi_b * (a * b**2 - (gamma + kappa) * b) * dx
        )

        # Setup solver objects
        nlvp_a = NonlinearVariationalProblem(F_a, a)
        nlvs_a = NonlinearVariationalSolver(nlvp_a, ad_block_tag="a")
        nlvp_b = NonlinearVariationalProblem(F_b, b)
        nlvs_b = NonlinearVariationalSolver(nlvp_b, ad_block_tag="b")

        # Time integrate from t_start to t_end
        tp = mesh_seq.time_partition
        t_start, t_end = tp.subintervals[index]
        dt = tp.timesteps[index]
        t = t_start
        while t < t_end - 0.5 * dt:
            nlvs_a.solve()
            nlvs_b.solve()
            yield

            a_.assign(a)
            b_.assign(b)
            t += dt

    return solver


# Let's consider the same QoI, time partition, and mesh sequence as in the previous
# demo, so that the outputs can be straightforwardly compared. ::


def get_qoi(mesh_seq, index):
    def qoi():
        a = mesh_seq.fields["a"][0]
        b = mesh_seq.fields["b"][0]
        return a * b**2 * dx

    return qoi


end_time = 2000.0
dt = [0.0001, 0.001, 0.01, 0.1, (end_time - 1) / end_time]
num_subintervals = 5
dt_per_export = [10, 9, 9, 9, 10]
time_partition = TimePartition(
    end_time,
    num_subintervals,
    dt,
    field_names,
    num_timesteps_per_export=dt_per_export,
    subintervals=[
        (0.0, 0.001),
        (0.001, 0.01),
        (0.01, 0.1),
        (0.1, 1.0),
        (1.0, end_time),
    ],
)

mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_function_spaces=get_function_spaces,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver,
    get_qoi=get_qoi,
    qoi_type="end_time",
)
solutions = mesh_seq.solve_adjoint()

solutions.export(
    "gray_scott_split/solutions.pvd",
    export_field_types=["forward", "adjoint"],
    initial_condition=mesh_seq.get_initial_condition(),
)

# This tutorial can be dowloaded as a `Python script <gray_scott_split.py>`__.
