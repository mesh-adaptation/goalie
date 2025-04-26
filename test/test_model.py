"""
Testing for the Model objects.
"""

import unittest

from firedrake import (
    Function,
    FunctionSpace,
    UnitSquareMesh,
)
from parameterized import parameterized

from goalie.field import Field
from goalie.mesh_seq import MeshSeq
from goalie.model import Model
from goalie.time_partition import TimePartition


class BaseClasses:
    """
    Base classes for Model unit testing.
    """

    class ModelTestCase(unittest.TestCase):
        """
        Test case with a simple setUp method and model constructor.
        """

        def setUp(self):
            field = Field("field", family="Real")
            time_partition = TimePartition(1.0, 2, [0.5, 0.5], field)
            mesh_seq = MeshSeq([UnitSquareMesh(1, 1)])
            f_space = FunctionSpace(mesh_seq[0], "CG", 1)
            self.outputs_consistent_args = (
                time_partition,
                mesh_seq,
                {"field": (Function(f_space), Function(f_space))},
                {"field": f_space},
            )


class TestExceptions(BaseClasses.ModelTestCase):
    """
    Unit tests for exceptions raised by :class:`Model`.
    """

    @parameterized.expand(["get_solver", "get_qoi"])
    def test_notimplemented_error(self, method):
        model = Model()
        with self.assertRaises(NotImplementedError) as cm:
            getattr(model, method)(0, *self.outputs_consistent_args)
        self.assertEqual(
            str(cm.exception), f"Model Model is missing the '{method}' method."
        )

    def test_return_dict_error(self):
        model = Model()
        model.get_initial_condition = lambda *_: 0
        with self.assertRaises(AssertionError) as cm:
            model._outputs_consistent(*self.outputs_consistent_args)
        msg = "get_initial_condition should return a dict"
        self.assertEqual(str(cm.exception), msg)

    def test_missing_field_error(self):
        model = Model()
        model.get_initial_condition = lambda *_: {}
        with self.assertRaises(AssertionError) as cm:
            model._outputs_consistent(*self.outputs_consistent_args)
        msg = "missing fields {'field'} in get_initial_condition"
        self.assertEqual(str(cm.exception), msg)

    def test_unexpected_field_error(self):
        model = Model()
        model.get_initial_condition = lambda *_: {"field": None, "extra_field": None}
        with self.assertRaises(AssertionError) as cm:
            model._outputs_consistent(*self.outputs_consistent_args)
        msg = "unexpected fields {'extra_field'} in get_initial_condition"
        self.assertEqual(str(cm.exception), msg)

    def test_solver_generator_error(self):
        model = Model()
        model.get_solver = lambda *_: {}
        with self.assertRaises(AssertionError) as cm:
            model._outputs_consistent(*self.outputs_consistent_args)
        self.assertEqual(str(cm.exception), "get_solver should yield")
