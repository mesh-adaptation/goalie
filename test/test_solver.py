"""
Testing for the solver objects.
"""

import unittest

from firedrake import (
    Function,
    FunctionSpace,
    UnitSquareMesh,
)
from parameterized import parameterized

from goalie.mesh_seq import MeshSeq
from goalie.solver import Solver
from goalie.time_partition import TimeInterval


class BaseClasses:
    """
    Base classes for solver unit testing.
    """

    class SolverTestCase(unittest.TestCase):
        """
        Test case with a simple setUp method and mesh constructor.
        """

        def setUp(self):
            time_interval = TimeInterval(1.0, [0.5], ["field"])
            mesh_seq = MeshSeq([UnitSquareMesh(1, 1)])
            self.solver = Solver(time_interval, mesh_seq)


class TestExceptions(BaseClasses.SolverTestCase):
    """
    Unit tests for exceptions raised by :class:`Solver`.
    """

    @parameterized.expand(["get_function_spaces", "get_solver"])
    def test_notimplemented_error(self, function_name):
        with self.assertRaises(NotImplementedError) as cm:
            if function_name == "get_function_spaces":
                getattr(self.solver, function_name)(self.solver.meshes[0])
            else:
                getattr(self.solver, function_name)()
        self.assertEqual(
            str(cm.exception), f"Solver Solver is missing the '{function_name}' method."
        )

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_return_dict_error(self, method):
        kwargs = {method: lambda *_: 0}
        setattr(self.solver, method, kwargs[method])
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        self.assertEqual(str(cm.exception), f"{method} should return a dict")

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_missing_field_error(self, method):
        kwargs = {method: lambda *_: {}}
        setattr(self.solver, method, kwargs[method])
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        msg = "missing fields {'field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_unexpected_field_error(self, method):
        kwargs = {method: lambda *_: {"field": None, "extra_field": None}}
        setattr(self.solver, method, kwargs[method])
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        msg = "unexpected fields {'extra_field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    def test_solver_generator_error(self):
        mesh = self.solver.meshes[0]
        f_space = FunctionSpace(mesh, "CG", 1)
        kwargs = {
            "get_function_spaces": lambda _: {"field": f_space},
            "get_initial_condition": lambda *_: {"field": Function(f_space)},
            "get_solver": lambda _: lambda *_: {},
        }
        for key, value in kwargs.items():
            setattr(self.solver, key, value)
        with self.assertRaises(AssertionError) as cm:
            self.solver._outputs_consistent()
        self.assertEqual(str(cm.exception), "solver should yield")
