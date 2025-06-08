"""
Unit tests for the optimisation module.
"""

import unittest
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import VertexOnlyMesh
from firedrake.utility_meshes import UnitIntervalMesh
from goalie.optimisation import OptimisationProgress


class TestOptimisationProgress(unittest.TestCase):
    """
    Unit tests for the :class:`goalie.optimisation.OptimisationProgress` class.
    """

    def setUp(self):
        self.progress = OptimisationProgress()

    def test_initial_state(self):
        self.assertEqual(self.progress["qoi"], [])
        self.assertEqual(self.progress["control"], [])
        self.assertEqual(self.progress["gradient"], [])

    def test_reset(self):
        self.progress["qoi"].append(1.0)
        self.progress["control"].append(2.0)
        self.progress["gradient"].append(3.0)
        self.progress.reset()
        self.test_initial_state()

    def test_convert_to_float(self):
        mesh = VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])
        R = FunctionSpace(mesh, "R", 0)
        self.progress["qoi"] = [Function(R).assign(1.0)]
        self.progress["control"] = [Function(R).assign(2.0)]
        self.progress["gradient"] = [Function(R).assign(3.0)]
        self.progress.convert_to_float()
        self.assertEqual(self.progress["qoi"], [1.0])
        self.assertEqual(self.progress["control"], [2.0])
        self.assertEqual(self.progress["gradient"], [3.0])


if __name__ == "__main__":
    unittest.main()
