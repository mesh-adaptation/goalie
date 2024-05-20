"""
Checks that all demo scripts run.
"""

import glob
import os
import shutil
from os.path import splitext

import pytest

cwd = os.path.abspath(os.path.dirname(__file__))
demo_dir = os.path.abspath(os.path.join(cwd, "..", "demos"))
all_demos = glob.glob(os.path.join(demo_dir, "*.py"))

# Modifications dictionary to cut down run time of demos:
# - Each key is the name of a demo script to be modified
# - Each value is a dictionary where:
# -- The key is the original string or block of code to be replaced
# -- The value is the replacement string (use "" to remove the original code)
modifications = {
    "burgers_hessian.py": {"'maxiter': 35": "'maxiter': 3"},
    "gray_scott.py": {
        "end_time = 2000.0": "end_time = 10.0",
        """
        ic = mesh_seq.get_initial_condition()
        for field, sols in solutions.items():
            fwd_outfile = VTKFile(f"gray_scott/{field}_forward.pvd")
            adj_outfile = VTKFile(f"gray_scott/{field}_adjoint.pvd")
            fwd_outfile.write(*ic[field].subfunctions)
            for i in range(num_subintervals):
                for sol in sols["forward"][i]:
                    fwd_outfile.write(*sol.subfunctions)
                for sol in sols["adjoint"][i]:
                    adj_outfile.write(*sol.subfunctions)
            adj_end = Function(ic[field]).assign(0.0)
            adj_outfile.write(*adj_end.subfunctions)
        """: "",
    },
    "gray_scott_split.py": {
        "end_time = 2000.0": "end_time = 10.0",
        """
        ic = mesh_seq.get_initial_condition()
        for field, sols in solutions.items():
            fwd_outfile = VTKFile(f"gray_scott/{field}_forward.pvd")
            adj_outfile = VTKFile(f"gray_scott/{field}_adjoint.pvd")
            fwd_outfile.write(*ic[field].subfunctions)
            for i in range(num_subintervals):
                for sol in sols["forward"][i]:
                    fwd_outfile.write(*sol.subfunctions)
                for sol in sols["adjoint"][i]:
                    adj_outfile.write(*sol.subfunctions)
            adj_end = Function(ic[field]).assign(0.0)
            adj_outfile.write(*adj_end.subfunctions)
        """: "",
    },
    "point_discharge2d-hessian.py": {"'maxiter': 35": "'maxiter': 3"},
    "point_discharge2d-goal_oriented.py": {"'maxiter': 35": "'maxiter': 3"},
    "solid_body_rotation.py": {
        "end_time = 2 * pi": "end_time = pi / 4",
        """
        for field, sols in solutions.items():
            fwd_outfile = VTKFile(f"solid_body_rotation/{field}_forward.pvd")
            adj_outfile = VTKFile(f"solid_body_rotation/{field}_adjoint.pvd")
            for i in range(len(mesh_seq)):
                for sol in sols["forward"][i]:
                    fwd_outfile.write(sol)
                for sol in sols["adjoint"][i]:
                    adj_outfile.write(sol)
        """: "",
    },
}


@pytest.fixture(params=all_demos, ids=lambda x: splitext(x.split("demos/")[-1])[0])
def demo_file(request):
    return os.path.abspath(request.param)


def test_demos(demo_file, tmpdir, monkeypatch):
    assert os.path.isfile(demo_file), f"Demo file '{demo_file}' not found."

    # Copy mesh files
    source = os.path.dirname(demo_file)
    for f in glob.glob(os.path.join(source, "*.msh")):
        shutil.copy(f, str(tmpdir))

    # Change working directory to temporary directory
    monkeypatch.chdir(tmpdir)

    # Read the original demo script
    with open(demo_file, "r") as f:
        demo_contents = f.read()

    # Modify the demo script to shorten run time
    demo_name = os.path.basename(demo_file)
    if demo_name in modifications:
        for original, replacement in modifications[demo_name].items():
            demo_contents = demo_contents.replace(original, replacement)

    # Execute the modified demo as a standalone script
    context = {
        "__file__": demo_file,
        "__name__": "__main__",
        "__package__": None,
    }
    exec(demo_contents, context)

    # Clean up plots
    for ext in ("jpg", "pvd", "vtu"):
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
