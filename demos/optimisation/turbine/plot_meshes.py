import argparse
import matplotlib.pyplot as plt
import os

from firedrake.checkpointing import CheckpointFile
from firedrake.pyplot import triplot

from utils import get_latest_experiment_id

# Add argparse for command-line arguments
parser = argparse.ArgumentParser(description="Plot meshes from a given simulation.")
parser.add_argument("--n", type=int, default=0, help="Initial mesh resolution.")
parser.add_argument(
    "--anisotropic",
    action="store_true",
    help="Use anisotropic adaptation (default: False).",
)
args = parser.parse_args()

# Use parsed arguments
n = args.n
anisotropic = args.anisotropic
aniso_str = "aniso" if anisotropic else "iso"
config_str = f"{aniso_str}_n{n}"

# Determine the experiment_id and get associated directory
experiment_id = get_latest_experiment_id()
print(f"Experiment ID: {experiment_id}")
data_dir = os.path.join("data", experiment_id)
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

# Write each intermediate adapted mesh
file_exists = True
iteration = 1
while file_exists:
    checkpoint_filename = os.path.join(data_dir, f"{config_str}_mesh{iteration}.h5")
    file_exists = os.path.exists(checkpoint_filename)
    if not file_exists:
        break

    with CheckpointFile(checkpoint_filename, "w") as chk:
        # FIXME: Topology name is wrong
        mesh = chk.load_mesh()

    fig, axes = plt.subplots(figsize=(12, 5))
    interior_kw = {"edgecolor": "k", "linewidth": 0.5}
    triplot(mesh, axes=axes, interior_kw=interior_kw)
    axes.set_title(f"Mesh at iteration {iteration}")
    fig.savefig(f"{plot_dir}/{config_str}_mesh{iteration}.jpg")
    plt.close()

    iteration += 1
