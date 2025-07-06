import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot progress of controls and QoIs.")
parser.add_argument("--n", type=int, required=True, help="Initial mesh resolution.")
parser.add_argument(
    "--approach",
    type=str,
    choices=["fixed_mesh", "goal_oriented"],
    required=True,
    help="Approach type (fixed_mesh or goal_oriented).",
)
parser.add_argument(
    "--anisotropic", action="store_true", help="Flag to indicate anisotropic mode."
)
args = parser.parse_args()

# Use parsed arguments
n = args.n
approach = args.approach
anisotropic = args.anisotropic
aniso_str = "aniso_" if anisotropic else "iso_"

if approach == "goal_oriented":
    approach += "_" + aniso_str
controls = np.load(f"{approach}_{n}_control.npy")
qois = np.load(f"{approach}_{n}_qoi.npy")

fig, axes = plt.subplots()
axes.plot(controls, "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("Control")
axes.grid(True)
plt.savefig(f"{approach}_{n}_control.jpg", bbox_inches="tight")

fig, axes = plt.subplots()
axes.plot(qois, "--x")
axes.set_xlabel("Iteration")
axes.set_ylabel("QoI")
axes.grid(True)
plt.savefig(f"{approach}_{n}_qoi.jpg", bbox_inches="tight")
