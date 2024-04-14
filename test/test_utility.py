"""
Test utility functions.
"""

from firedrake import *
from goalie import *
from utility import uniform_mesh
import os
import pathlib
import shutil
import unittest


pointwise_norm_types = [["l1"], ["l2"], ["linf"]]
integral_scalar_norm_types = [["L1"], ["L2"], ["L4"], ["H1"], ["HCurl"]]
scalar_norm_types = pointwise_norm_types + integral_scalar_norm_types

# ---------------------------
# standard tests for pytest
# ---------------------------


class TestEffectivityIndex(unittest.TestCase):
    """
    Unit tests for :func:`effectivity_index`.
    """

    def test_error_indicator_not_function_error(self):
        with self.assertRaises(ValueError) as cm:
            effectivity_index(1.0, 1.0)
        msg = "Error indicator must return a Function."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_degree(self):
        mesh = uniform_mesh(2, 1)
        ei = Function(FunctionSpace(mesh, "DG", 1))
        with self.assertRaises(ValueError) as cm:
            effectivity_index(ei, 1.0)
        msg = "Error indicator must be P0."
        self.assertEqual(str(cm.exception), msg)

    def test_accuracy(self):
        mesh = uniform_mesh(2, 1)  # Two elements
        ei = Function(FunctionSpace(mesh, "DG", 0))
        ei.assign(1.0)
        self.assertAlmostEqual(effectivity_index(ei, 1.0), 2.0)


def test_create_directory():
    """
    Test that :func:`create_directory` works as expected.
    """
    pwd = os.path.dirname(__file__)
    fpath = os.path.join(pwd, "tmp")

    # Delete the directory if it already exists
    #   FIXME: Why does it already exist on the CI platform?
    if os.path.exists(fpath):
        shutil.rmtree(fpath)
    assert not os.path.exists(fpath)

    # Create the new directory
    new_fpath = create_directory(fpath)
    assert os.path.exists(new_fpath)
    assert new_fpath == fpath

    # Check create_directory works when it already exists
    new_fpath = create_directory(fpath)
    assert os.path.exists(new_fpath)
    assert new_fpath == fpath

    # Remove the directory
    try:
        pathlib.Path(fpath).rmdir()
    except OSError:
        ls = ", ".join(os.listdir(fpath))
        raise OSError(f"Can't remove {fpath} because it isn't empty. Contents: {ls}.")


if __name__ == "__main__":
    unittest.main()
