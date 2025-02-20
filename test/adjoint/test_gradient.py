"""
Testing for gradient computation.
"""

import unittest

from firedrake import UnitSquareMesh

from goalie.adjoint import AdjointMeshSeq
from goalie.time_partition import TimePartition


class TestGradient(unittest.TestCase):
    """
    Unit tests for gradient computation.
    """

    def setUp(self):
        end_time = 1.0
        num_subintervals = 2
        dt = end_time / num_subintervals
        self.time_partition = TimePartition(end_time, num_subintervals, dt, "field")
        self.meshes = [UnitSquareMesh(4, 4) for _ in range(num_subintervals)]

    def test_attribute_error(self):
        mesh_seq = AdjointMeshSeq(
            self.time_partition,
            self.meshes,
            qoi_type="end_time",
        )
        with self.assertRaises(AttributeError) as cm:
            _ = mesh_seq.gradient
        msg = (
            "To compute the gradient, pass compute_gradient=True to the solve_adjoint"
            " method."
        )
        self.assertEqual(str(cm.exception), msg)
