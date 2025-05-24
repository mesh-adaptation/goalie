"""
Testing for the Solver objects.
"""

import unittest

from firedrake import (
    UnitSquareMesh,
)
from parameterized import parameterized

from goalie.field import Field
from goalie.mesh_seq import MeshSeq
from goalie.solver import Solver
from goalie.time_partition import TimePartition


class BaseClasses:
    """
    Base classes for Solver unit testing.
    """

    class SolverTestCase(unittest.TestCase):
        """
        Test case with a simple setUp method.
        """

        def setUp(self):
            field = Field("field", family="Real")
            time_partition = TimePartition(1.0, 1, 0.5, field)
            mesh_seq = MeshSeq([UnitSquareMesh(1, 1)])
            self.solver = Solver(time_partition, mesh_seq)


class TestExceptions(BaseClasses.SolverTestCase):
    """
    Unit tests for exceptions raised by :class:`Solver`.
    """

    @parameterized.expand(["get_solver", "get_qoi"])
    def test_notimplemented_error(self, method):
        with self.assertRaises(NotImplementedError) as cm:
            getattr(self.solver, method)(0)
        self.assertEqual(
            str(cm.exception), f"Solver Solver is missing the '{method}' method."
        )

    def test_return_dict_error(self):
        self.solver.get_initial_condition = lambda *_: 0
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        msg = "get_initial_condition should return a dict"
        self.assertEqual(str(cm.exception), msg)

    def test_missing_field_error(self):
        self.solver.get_initial_condition = lambda *_: {}
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        msg = "missing fields {'field'} in get_initial_condition"
        self.assertEqual(str(cm.exception), msg)

    def test_unexpected_field_error(self):
        self.solver.get_initial_condition = lambda *_: {
            "field": None,
            "extra_field": None,
        }
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        msg = "unexpected fields {'extra_field'} in get_initial_condition"
        self.assertEqual(str(cm.exception), msg)

    def test_solver_generator_error(self):
        self.solver.get_solver = lambda *_: {}
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        self.assertEqual(str(cm.exception), "get_solver should yield")
