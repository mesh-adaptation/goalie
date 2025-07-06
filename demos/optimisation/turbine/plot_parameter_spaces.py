import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from setup import qoi_scaling

scaling = 1e-6 / qoi_scaling

# Plot the trajectory with the maximum QoIs highlighted
fig, axes = plt.subplots()
for n in range(3):
    sampled_controls = np.load(f"controls_{n}.npy")
    sampled_qois = -np.load(f"qois_{n}.npy") * scaling

    # Perform cubic interpolation
    cubic_spline = CubicSpline(sampled_controls, sampled_qois)

    # Find the derivative of the cubic spline
    derivative = cubic_spline.derivative()

    # Find the critical points (where the derivative is zero)
    critical_controls = derivative.roots()

    # Evaluate the cubic spline at the critical points
    critical_qois = cubic_spline(critical_controls)

    # Find the maximum QoI and its corresponding control
    max_index = np.argmax(critical_qois)
    max_control = critical_controls[max_index]
    max_qoi = critical_qois[max_index]
    print(f"Maximum QoI: {max_qoi} at Control: {max_control}")

    axes.plot(
        sampled_controls, sampled_qois, "--x", color=f"C{n}", label="Sampled data"
    )
    axes.plot(max_control, max_qoi, "o", color=f"C{n}", label="Maximum value")
axes.set_xlabel(r"Control turbine position [$\mathrm{m}$]")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"progress_parameter_spaces.png", bbox_inches="tight")
