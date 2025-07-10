import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import CubicSpline

from setup import qoi_scaling

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(
    description="Plot progress of controls and QoIs on the same axis."
)
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
args = parser.parse_args()

n = args.n
scaling = 1e-6 / qoi_scaling

sampled_controls = np.load(f"outputs/fixed_mesh_{n}/sampled_controls.npy")
sampled_qois = -np.load(f"outputs/fixed_mesh_{n}/sampled_qois.npy") * scaling

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

fixed_mesh_controls = np.load(f"outputs/fixed_mesh_{n}/control.npy")
fixed_mesh_qois = -np.load(f"outputs/fixed_mesh_{n}/qoi.npy") * scaling

# Plot the trajectory with the maximum QoI highlighted
fig, axes = plt.subplots()
axes.plot(sampled_controls, sampled_qois, "--x", label="Sampled data")
axes.plot(fixed_mesh_controls, fixed_mesh_qois, "--^", label="Fixed mesh")
axes.plot(max_control, max_qoi, "o", label="Maximum value")
axes.set_xlabel(r"Control turbine position [$\mathrm{m}$]")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
axes.legend()
plt.savefig(f"plots/fixed_mesh_{n}/progress_parameter_space.png", bbox_inches="tight")
