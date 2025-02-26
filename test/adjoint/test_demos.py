"""
Checks that all demo scripts run.
"""

import glob
import os
import re
import shutil
from os.path import splitext

import pytest

from goalie.log import WARNING, set_log_level

cwd = os.path.abspath(os.path.dirname(__file__))
demo_dir = os.path.abspath(os.path.join(cwd, "..", "..", "demos"))
all_demos = glob.glob(os.path.join(demo_dir, "*.py"))

# Modifications dictionary to cut down run time of demos:
# - Each key is the name of a demo script to be modified
# - Each value is a dictionary where:
# -- The key is the original string or regex pattern that we wish to replace
# -- The value is the replacement string (use "" to remove the original code)
modifications = {
    "burgers-hessian.py": {""""maxiter": 35""": """"maxiter": 3"""},
    "gray_scott.py": {
        "end_time = 2000.0": "end_time = 10.0",
        r"solutions\.export\([\s\S]*?\)\s*,?\s*\)?\n?": "",
    },
    "gray_scott_split.py": {
        "end_time = 2000.0": "end_time = 10.0",
        r"solutions\.export\([\s\S]*?\)\s*,?\s*\)?\n?": "",
    },
    "point_discharge2d-hessian.py": {""""maxiter": 35""": """"maxiter": 3"""},
    "point_discharge2d-goal_oriented.py": {""""maxiter": 35""": """"maxiter": 3"""},
    "solid_body_rotation.py": {
        r"solutions\.export\((.*?)\)": "",
    },
    "burgers-goal_oriented.py": {
        """"maxiter": 35""": """"maxiter": 2""",
        "end_time = 0.5": "end_time = 0.125",
    },
}


@pytest.fixture(params=all_demos, ids=lambda x: splitext(x.split("demos/")[-1])[0])
def demo_file(request):
    return os.path.abspath(request.param)


def test_modifications_demo_exists():
    """
    Check that all demos named in the modifications dictionary exist in the 'demos' dir.
    """
    for demo_name in modifications.keys():
        demo_fpath = os.path.join(demo_dir, demo_name)
        assert demo_fpath in all_demos, f"Error: Demo '{demo_name}' not found."


def test_modifications_original_exists():
    """
    Check that all 'original' code snippets in the modifications dictionary exist in the
    corresponding demo scripts.
    """
    for demo_name, changes in modifications.items():
        demo_path = os.path.join(demo_dir, demo_name)
        with open(demo_path, "r") as file:
            demo_content = file.read()
            for original in changes.keys():
                assert re.search(
                    original, demo_content, re.DOTALL
                ), f"Error: '{original}' not found in '{demo_name}'."


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
            demo_contents = re.sub(
                original, replacement, demo_contents, flags=re.DOTALL
            )

    # Execute the modified demo as a standalone script in a clean namespace
    exec_namespace = {
        "__file__": demo_file,
        "__name__": "__main__",
        "__package__": None,
    }
    exec(demo_contents, exec_namespace)

    # Reset log level
    set_log_level(WARNING)

    # Clean up plots
    for ext in ("jpg", "pvd", "vtu"):
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
