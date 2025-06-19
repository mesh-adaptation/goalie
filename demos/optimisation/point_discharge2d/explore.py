# TODO: text

import matplotlib.pyplot as plt
from firedrake import *
from goalie import *
from firedrake.pyplot import *

from setup import *

# Consider a relatively fine uniform mesh
n = 5
mesh = RectangleMesh(100 * n, 20 * n, 50, 10)

# Explore the parameter space and compute the corresponding cost function values
controls = np.linspace(1.0, 9.0, 101)
qois = []
for control in controls:
    get_ic = lambda *args: get_initial_condition(*args, initial_control=control)

    time_partition = TimeInstant(fields)
    mesh_seq = AdjointMeshSeq(
        time_partition,
        mesh,
        get_initial_condition=get_ic,
        get_solver=get_solver,
        get_qoi=get_qoi,
        qoi_type="steady",
    )

    mesh_seq.get_checkpoints(run_final_subinterval=True)
    J = -mesh_seq.J
    print(f"control={control:6.4f}, qoi={J:11.4e}")
    qois.append(J)

# Save the trajectory to file
np.save(f"controls_{n}.npy", controls)
np.save(f"qois_{n}.npy", qois)

# Plot the trajectory
fig, axes = plt.subplots()
axes.plot(controls, qois, "--x")
axes.set_xlabel(r"Control, $y_c$")
axes.set_ylabel(r"Cost, $\int_{\Gamma}c\;\mathrm{d}s$")
axes.grid(True)
plt.savefig(f"parameter_space_{n}.png", bbox_inches="tight")
