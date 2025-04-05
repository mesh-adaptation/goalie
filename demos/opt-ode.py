# ODE-constrained optimisation
# ============================
#
# Goalie was designed primarily with goal-oriented mesh adaptation in mind and
# implements the adjoint methods required to compute goal-oriented error estimates on
# sequences of (adapted) meshes. However, it can also be used for ODE- and
# PDE-constrained optimisation problems. This demo illustrates how to use Goalie to
# solve a simple ODE-constrained optimisation problem.
#
# Recall the scalar, linear ODE from `the first ODE demo <./ode.py.html>`__:
#
# .. math::
#    \frac{\mathrm{d}u}{\mathrm{d}t} = u,\quad u(0) = 1,
#
# which we solve for :math:`u(t)` over the time period :math:`t\in(0,1]`. In this demo,
# we will optimise the :math:`theta` parameter in the :math:`theta` timestepping scheme
# outlined in the `ODE demo <./ode.py.html>`__ such that the $L^2$ error between the
# numerical solution and the analytical solution at the final time is minimised.
#
# Begin with some standard imports. ::

import matplotlib.pyplot as plt
from firedrake import *

from goalie_adjoint import *

# Copy over much of the code from the `ODE demo <./ode.py.html>`__ demo, with some key
# differences:
#
# 1. We define the `theta` parameter as a :class:`~.Field` in R-space and set it to have
#    the attributes `unsteady=False` and `solved_for=False`. This means that this
#    control field doesn't change with time and is not solved for as part of the forward
#    problem. However, we can still compute gradients with respect to it.
# 2. The `get_initial_condition` function needs to a set an initial value for `theta`.
#    We set this to zero, which implies that we start with Forward Euler.
# 3. We also reconfigure the `get_solver_theta` function so that it reads the `theta`
#    field from the `mesh_seq.field_functions` dictionary and uses it in the Forward
#    Euler scheme. ::

mesh = VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])
fields = [
    Field("u", mesh=mesh, family="Real", degree=0),
    Field(
        "theta", mesh=mesh, family="Real", degree=0, unsteady=False, solved_for=False
    ),
]
dt = 0.2
end_time = 1
time_partition = TimeInterval(end_time, dt, fields)


def get_initial_condition(mesh_seq):
    return {
        "u": Function(mesh_seq.function_spaces["u"][0]).assign(1.0),
        "theta": Function(mesh_seq.function_spaces["theta"][0]).assign(0.0),
    }


def get_solver_theta(mesh_seq):
    def solver(index):
        u, u_ = mesh_seq.field_functions["u"]
        theta = mesh_seq.field_functions["theta"]
        R = mesh_seq.function_spaces["u"][index]
        tp = mesh_seq.time_partition
        dt = Function(R).assign(tp.timesteps[index])
        v = TestFunction(R)

        F = (u - u_ - dt * (theta * u + (1 - theta) * u_)) * v * dx

        sp = {"ksp_type": "preonly", "pc_type": "jacobi"}
        t_start, t_end = tp.subintervals[index]
        dt = tp.timesteps[index]
        t = t_start
        while t < t_end - 1.0e-05:
            solve(F == 0, u, solver_parameters=sp, ad_block_tag="u")
            yield

            u_.assign(u)
            t += dt

    return solver


# In order to perform ODE-constrained optimisation, we need to define a quantity of
# interest. For this problem, we have an analytical solution for the ODE ($u(t) = e^t$),
# so we can artificially set the quantity of interest to be the $L^2$ error between the
# numerical solution and the analytical solution at the final time. ::


def get_qoi(mesh_seq, index):
    def end_time_qoi():
        sol = exp(1.0)
        u = mesh_seq.field_functions["u"][0]
        return inner(u - sol, u - sol) * dx

    return end_time_qoi


mesh_seq = AdjointMeshSeq(
    time_partition,
    mesh,
    get_initial_condition=get_initial_condition,
    get_solver=get_solver_theta,
    get_qoi=get_qoi,
    qoi_type="end_time",
)

# Print initial field values. ::

ics = get_initial_condition(mesh_seq)
for fieldname, field in ics.items():
    if isinstance(field, tuple):
        print(f"{fieldname}_0 = {float(field[0]):.4e}")
    else:
        print(f"{fieldname}_0 = {float(field):.4e}")

# We see that 'u' and 'theta' are initialised to 1.0 and 0.0 respectively:
#
# .. code-block:: none
#
#     u_0 = 1.0000e+00
#     theta_0 = 0.0000e+00
#
# Before running the optimisation, we solve the forward and adjoint problems to
# visualise the trajectory and the initial gradient values. ::

solutions = mesh_seq.solve_adjoint(compute_gradient=True)

# Print QoI value and the initial gradient values. ::

J = mesh_seq.J
print(f"J = {J:.4e}")
for fieldname, gradient in mesh_seq.gradient.items():
    print(f"dJ/d{fieldname} = {float(gradient):.4e}")

# We have:
#
# .. code-block:: none
#
#     J = 5.2882e-02
#     dJ/du = -1.1444e+00
#     dJ/dtheta = -1.9074e-01
#
# Plot the trajectory and compare it against the analytical solution. ::

forward_euler_trajectory = [float(get_initial_condition(mesh_seq)["u"])]
forward_euler_trajectory += [
    float(sol) for subinterval in solutions["u"]["forward"] for sol in subinterval
]
times = np.arange(0, 1.01, dt)

fig, axes = plt.subplots()
axes.plot(times, np.exp(times), "--x", label="Analytical solution")
axes.plot(times, forward_euler_trajectory, "--+", label="Forward Euler")
axes.set_xlabel(r"Time, $t$")
axes.set_ylabel(r"$u$")
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig("opt-ode_forward_euler.jpg", bbox_inches="tight")

# .. figure:: opt-ode_forward_euler.jpg
#    :figwidth: 70%
#    :align: center
#
# This is just the same plot that we saw in the `ODE demo <./ode.py.html>`__ demo for
# Forward Euler.
#
# Specify parameters for the optimisation: learning rate / step length of 0.5 and
# maximum number of iterations of 100. Print the parameters to see what the other
# default values are. ::

parameters = OptimisationParameters({"lr": 0.5, "maxiter": 100})
print(parameters)

# Run the optimisation using the gradient descent method. Here we specify that the
# control to be optimised is 'theta'. It would also be valid to specify 'u' as the
# control, but this would attempt to optimise the initial condition for 'u', which is
# not want we want here. ::

optimiser = QoIOptimiser(mesh_seq, "theta", parameters, method="gradient_descent")
optimiser.minimise()

# This should give output similar to the following:
#
# .. code-block:: none
#
#     it= 1, u=9.5370e-02, J=5.2882e-02, dJ=-1.9074e-01, lr=5.0000e-01
#     it= 2, u=1.7806e-01, J=3.5864e-02, dJ=-1.6537e-01, lr=5.0000e-01
#     it= 3, u=2.4772e-01, J=2.3239e-02, dJ=-1.3933e-01, lr=5.0000e-01
#     it= 4, u=3.0474e-01, J=1.4394e-02, dJ=-1.1404e-01, lr=5.0000e-01
#     it= 5, u=3.5012e-01, J=8.5440e-03, dJ=-9.0767e-02, lr=5.0000e-01
#     it= 6, u=3.8534e-01, J=4.8800e-03, dJ=-7.0423e-02, lr=5.0000e-01
#     it= 7, u=4.1205e-01, J=2.6961e-03, dJ=-5.3435e-02, lr=5.0000e-01
#     it= 8, u=4.3195e-01, J=1.4492e-03, dJ=-3.9798e-02, lr=5.0000e-01
#     it= 9, u=4.4655e-01, J=7.6204e-04, dJ=-2.9202e-02, lr=5.0000e-01
#     it=10, u=4.5714e-01, J=3.9396e-04, dJ=-2.1180e-02, lr=5.0000e-01
#     it=11, u=4.6476e-01, J=2.0107e-04, dJ=-1.5228e-02, lr=5.0000e-01
#     it=12, u=4.7020e-01, J=1.0165e-04, dJ=-1.0877e-02, lr=5.0000e-01
#     it=13, u=4.7406e-01, J=5.1038e-05, dJ=-7.7323e-03, lr=5.0000e-01
#     it=14, u=4.7680e-01, J=2.5496e-05, dJ=-5.4779e-03, lr=5.0000e-01
#     it=15, u=4.7874e-01, J=1.2691e-05, dJ=-3.8712e-03, lr=5.0000e-01
#     it=16, u=4.8010e-01, J=6.3013e-06, dJ=-2.7309e-03, lr=5.0000e-01
#     it=17, u=4.8106e-01, J=3.1229e-06, dJ=-1.9241e-03, lr=5.0000e-01
#     it=18, u=4.8174e-01, J=1.5457e-06, dJ=-1.3545e-03, lr=5.0000e-01
#     it=19, u=4.8222e-01, J=7.6439e-07, dJ=-9.5288e-04, lr=5.0000e-01
#     it=20, u=4.8255e-01, J=3.7776e-07, dJ=-6.7006e-04, lr=5.0000e-01
#     it=21, u=4.8279e-01, J=1.8661e-07, dJ=-4.7104e-04, lr=5.0000e-01
#     it=22, u=4.8295e-01, J=9.2150e-08, dJ=-3.3106e-04, lr=5.0000e-01
#     it=23, u=4.8307e-01, J=4.5496e-08, dJ=-2.3264e-04, lr=5.0000e-01
#     it=24, u=4.8315e-01, J=2.2458e-08, dJ=-1.6346e-04, lr=5.0000e-01
#     it=25, u=4.8321e-01, J=1.1085e-08, dJ=-1.1485e-04, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=26, u=4.8325e-01, J=5.4709e-09, dJ=-8.0685e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=27, u=4.8328e-01, J=2.7000e-09, dJ=-5.6683e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=28, u=4.8330e-01, J=1.3324e-09, dJ=-3.9820e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=29, u=4.8331e-01, J=6.5753e-10, dJ=-2.7973e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=30, u=4.8332e-01, J=3.2447e-10, dJ=-1.9651e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=31, u=4.8333e-01, J=1.6012e-10, dJ=-1.3804e-05, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=32, u=4.8333e-01, J=7.9012e-11, dJ=-9.6971e-06, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=33, u=4.8334e-01, J=3.8989e-11, dJ=-6.8119e-06, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=34, u=4.8334e-01, J=1.9239e-11, dJ=-4.7852e-06, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=35, u=4.8334e-01, J=9.4938e-12, dJ=-3.3614e-06, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=36, u=4.8334e-01, J=4.6848e-12, dJ=-2.3613e-06, lr=5.0000e-01
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     it=37, u=4.8334e-01, J=2.3117e-12, dJ=-1.6587e-06, lr=5.0000e-01
#     Gradient convergence detected
#     WARNING AdjointMeshSeq: Zero QoI. Is it implemented as intended?
#     Optimised QoI: 1.1407e-12
#     Optimised control: 0.4833
#
# We find that the optimisation routine converges to a value just below 0.5, which can
# be viewed as a slight tweak to the Crank-Nicolson scheme seen previously.
#
# We can view the progress of the optimisation by plotting the control and QoI values
# as follows. ::

fig, axes = plt.subplots()
axes.plot(optimiser.progress["control"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig("opt-ode_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(optimiser.progress["qoi"], "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig("opt-ode_qoi.jpg", bbox_inches="tight")

# .. figure:: opt-ode_control.jpg
#   :figwidth: 70%
#   :align: center
#
# .. figure:: opt-ode_qoi.jpg
#   :figwidth: 70%
#   :align: center
#
# The control variable increases monotonically from zero to its final converged value,
# while the QoI decreases monotonically to a value close to zero, which is what we would
# expect from a QoI defined based on discretisation error.
#
# To visualise the optimised trajectory, we need to solve the problem again. ::

solutions = mesh_seq.solve_adjoint()
J = mesh_seq.J
print(f"Optimised QoI: {J:.4e}")
print(f"Optimised control: {float(mesh_seq.controls["theta"].tape_value()):.4f}")
opt_trajectory = [float(get_initial_condition(mesh_seq)["u"])]
opt_trajectory += [
    float(sol) for subinterval in solutions["u"]["forward"] for sol in subinterval
]
c_opt = optimiser.progress["control"][-1]
fig, axes = plt.subplots()
axes.plot(times, np.exp(times), "--x", label="Analytical solution")
axes.plot(times, forward_euler_trajectory, "--+", label=r"Forward Euler ($\theta=0$)")
axes.plot(times, opt_trajectory, "--+", label=rf"Optimised ($\theta={c_opt:.4f}$)")
axes.set_xlabel(r"Time, $t$")
axes.set_ylabel(r"$u$")
axes.grid(True)
axes.legend()
plt.tight_layout()
plt.savefig("opt-ode_optimised.jpg", bbox_inches="tight")

# .. figure:: opt-ode_optimised.jpg
#   :figwidth: 70%
#   :align: center
#
# In this final plot, we see that the optimised trajectory is right on top of the
# analytical solution, giving a significant improvement over the Forward Euler scheme.
#
# This demo can also be accessed as a `Python script <opt-ode.py>`__.
