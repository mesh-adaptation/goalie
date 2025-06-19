import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

n = 5
controls = np.load(f"controls_{n}.npy")
qois = np.load(f"qois_{n}.npy")

# Ensure controls and qois are sorted by controls for interpolation
sorted_indices = np.argsort(controls)

# Perform cubic interpolation
cubic_spline = CubicSpline(controls, qois)

# Example: Evaluate the cubic spline at new points
new_controls = np.linspace(controls.min(), controls.max(), 500)
interpolated_qois = cubic_spline(new_controls)

# Optional: Plot the original data and the interpolated curve
fig, axes = plt.subplots()
axes.plot(controls, qois, "o", label="Original Data")
axes.plot(new_controls, interpolated_qois, "-", label="Cubic Spline")
axes.set_xlabel(r"Control, $y_c$")
axes.set_ylabel(r"Cost, $\int_{\Gamma}c\;\mathrm{d}s$")
axes.grid(True)
axes.legend()
plt.savefig(f"cubic_spline_parameter_space_{n}.png", bbox_inches="tight")

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

# Plot the trajectory with the maximum QoI highlighted
fig, axes = plt.subplots()
axes.plot(controls, qois, "--x", label="Sampled data")
axes.plot(max_control, max_qoi, "o", label="Maximum value")
axes.set_xlabel(r"Control, $y_c$")
axes.set_ylabel(r"Cost, $\int_{\Gamma}c\;\mathrm{d}s$")
axes.grid(True)
axes.legend()
plt.savefig(f"parameter_space_{n}.png", bbox_inches="tight")
