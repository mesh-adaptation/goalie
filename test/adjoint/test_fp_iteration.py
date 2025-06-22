import abc
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import ufl
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.utility_meshes import UnitSquareMesh, UnitTriangleMesh
from parameterized import parameterized

from goalie.adjoint import AdjointSolver
from goalie.go_solver import GoalOrientedAdaptParameters, GoalOrientedSolver
from goalie.mesh_seq import MeshSeq
from goalie.options import AdaptParameters
from goalie.solver import Solver
from goalie.time_partition import TimeInstant, TimePartition


def constant_qoi(solver, index):
    R = FunctionSpace(solver.meshes[index], "R", 0)

    def qoi():
        return Function(R).assign(1) * ufl.dx

    return qoi


def oscillating_qoi(solver, index):
    R = FunctionSpace(solver.meshes[index], "R", 0)

    def qoi():
        return Function(R).assign(1 if solver.fp_iteration % 2 == 0 else 2) * ufl.dx

    return qoi


def oscillating_adaptor0(solver, *args):
    solver.meshes[0] = (
        UnitSquareMesh(1, 1) if solver.fp_iteration % 2 == 0 else UnitTriangleMesh()
    )
    return [False] * len(solver.meshes)


def oscillating_adaptor1(solver, *args):
    solver.meshes[1] = (
        UnitSquareMesh(1, 1) if solver.fp_iteration % 2 == 0 else UnitTriangleMesh()
    )
    return [False] * len(solver.meshes)


def empty_adaptor(*args):
    return [False]


class SolverBaseClass:
    """
    Base class for :meth:`fixed_point_iteration` unit tests.
    """

    @abc.abstractmethod
    def setUp(self):
        pass

    @abc.abstractmethod
    def set_values(self, mesh_seq, value):
        pass

    @abc.abstractmethod
    def check_convergence(self, mesh_seq):
        pass

    @property
    def default_kwargs(self):
        return {
            "time_partition": TimeInstant([]),
            "mesh_seq": MeshSeq([UnitTriangleMesh()]),
        }

    def solver(self, **kwargs):
        kw = self.default_kwargs
        kw.update(kwargs)
        solver = self.seq(kw.pop("time_partition"), kw.pop("mesh_seq"), **kw)
        solver.get_function_spaces = lambda _: {}
        solver.get_solver = lambda _: (yield from (None for _ in iter(int, 1)))
        solver.params = self.parameters
        return solver

    def test_convergence_noop(self):
        miniter = self.parameters.miniter
        solver = self.solver()
        solver.fixed_point_iteration(empty_adaptor, parameters=self.parameters)
        self.assertEqual(len(solver.meshes.element_counts), miniter + 1)
        self.assertTrue(np.allclose(solver.converged, True))
        self.assertTrue(np.allclose(solver.check_convergence, True))

    def test_noconvergence(self):
        maxiter = self.parameters.maxiter
        mesh_seq = self.solver()
        mesh_seq.fixed_point_iteration(oscillating_adaptor0, parameters=self.parameters)
        self.assertEqual(len(mesh_seq.meshes.element_counts), maxiter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_no_late_convergence(self):
        self.parameters.drop_out_converged = True
        solver = self.solver(
            time_partition=TimePartition(1.0, 2, [0.5, 0.5], []),
            mesh_seq=MeshSeq([UnitTriangleMesh(), UnitTriangleMesh()]),
        )
        with patch("goalie.go_solver.GoalOrientedSolver.forms") as mock_forms:
            mock_forms.return_value = MagicMock()
            solver.fixed_point_iteration(
                oscillating_adaptor0,
                parameters=self.parameters,
            )
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(solver.meshes.element_counts, expected)
        self.assertTrue(np.allclose(solver.converged, [False, False]))
        self.assertTrue(np.allclose(solver.check_convergence, [True, True]))

    @parameterized.expand([[True], [False]])
    def test_dropout(self, drop_out_converged):
        self.parameters.drop_out_converged = drop_out_converged
        solver = self.solver(
            time_partition=TimePartition(1.0, 2, [0.5, 0.5], []),
            mesh_seq=MeshSeq([UnitTriangleMesh(), UnitTriangleMesh()]),
        )
        with patch("goalie.go_solver.GoalOrientedSolver.forms") as mock_forms:
            mock_forms.return_value = MagicMock()
            solver.fixed_point_iteration(
                oscillating_adaptor1,
                parameters=self.parameters,
            )
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(solver.meshes.element_counts, expected)
        self.assertTrue(np.allclose(solver.converged, [True, False]))
        self.assertTrue(
            np.allclose(solver.check_convergence, [not drop_out_converged, True])
        )

    def test_update_params(self):
        def update_params(params, fp_iteration):
            params.element_rtol = fp_iteration

        self.parameters.miniter = self.parameters.maxiter
        self.parameters.element_rtol = 0.5
        solver = self.solver()
        solver.fixed_point_iteration(
            empty_adaptor, parameters=self.parameters, update_params=update_params
        )
        self.assertEqual(self.parameters.element_rtol + 1, 5)

    def test_convergence_lt_miniter(self):
        solver = self.solver()
        self.assertFalse(self.check_convergence(solver))
        self.assertFalse(solver.converged)

    def test_convergence_true(self):
        self.parameters.drop_out_converged = True
        solver = self.solver()
        self.set_values(solver, np.ones((solver.params.miniter + 1, 1)))
        self.assertTrue(self.check_convergence(solver))
        return solver

    def test_convergence_false(self):
        self.parameters.drop_out_converged = True
        solver = self.solver()
        values = np.ones((self.parameters.miniter + 1, 1))
        values[-1] = 2
        self.set_values(solver, values)
        self.assertFalse(self.check_convergence(solver))
        return solver

    def test_convergence_check_false(self):
        self.parameters.drop_out_converged = True
        solver = self.solver()
        self.set_values(solver, np.ones((solver.params.miniter + 1, 1)))
        solver.check_convergence[:] = True
        solver.check_convergence[-1] = False
        self.assertFalse(self.check_convergence(solver))


class TestSolver(unittest.TestCase, SolverBaseClass):
    """
    Unit tests for :meth:`MeshSeq.fixed_point_iteration`.
    """

    seq = Solver

    def setUp(self):
        self.parameters = AdaptParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def set_values(self, solver, value):
        solver.element_counts = value

    def check_convergence(self, solver):
        return solver.check_element_count_convergence()

    def test_converged_array_true(self):
        self.assertTrue(super().test_convergence_true().converged)

    def test_converged_array_false(self):
        self.assertFalse(super().test_convergence_false().converged)


class TestAdjointSolver(unittest.TestCase, SolverBaseClass):
    """
    Unit tests for :meth:`AdjointMeshSeq.fixed_point_iteration`.
    """

    seq = AdjointSolver

    def setUp(self):
        self.parameters = GoalOrientedAdaptParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    @property
    def default_kwargs(self):
        kw = super().default_kwargs
        kw["get_qoi"] = oscillating_qoi
        return kw

    def solver(self, **kwargs):
        kw = self.default_kwargs.copy()
        kw.update(kwargs)
        tp = kw["time_partition"]
        num_timesteps = 1 if tp is None else tp.num_timesteps
        kw["qoi_type"] = "steady" if num_timesteps == 1 else "end_time"
        return super().solver(**kw)

    def set_values(self, mesh_seq, value):
        mesh_seq.qoi_values = value

    def check_convergence(self, mesh_seq):
        return mesh_seq.check_qoi_convergence()


class TestGoalOrientedSolver(TestAdjointSolver):
    """
    Unit tests for :meth:`GoalOrientedSolver.fixed_point_iteration`.
    """

    seq = GoalOrientedSolver

    def set_values(self, mesh_seq, value):
        mesh_seq.estimator_values = value

    def check_convergence(self, mesh_seq):
        return mesh_seq.check_estimator_convergence()

    def test_convergence_criteria_all_false(self):
        self.parameters.convergence_criteria = "all"
        solver = self.solver(time_partition=TimePartition(1.0, 1, 0.5, []))
        with patch("goalie.go_solver.GoalOrientedSolver.forms") as mock_forms:
            mock_forms.return_value = MagicMock()
            solver.fixed_point_iteration(
                empty_adaptor,
                parameters=self.parameters,
            )
        self.assertTrue(np.allclose(solver.meshes.element_counts, 1))
        self.assertTrue(np.allclose(solver.converged, False))
        self.assertTrue(np.allclose(solver.check_convergence, True))

    def test_convergence_criteria_all_true(self):
        self.parameters.convergence_criteria = "all"
        solver = self.solver(
            time_partition=TimePartition(1.0, 1, 0.5, []),
            get_qoi=constant_qoi,
        )
        solver.error_estimate = MagicMock(return_value=1)
        with patch("goalie.go_solver.GoalOrientedSolver.forms") as mock_forms:
            mock_forms.return_value = MagicMock()
            solver.fixed_point_iteration(
                empty_adaptor,
                parameters=self.parameters,
            )
        self.assertTrue(np.allclose(solver.meshes.element_counts, 1))
        self.assertTrue(np.allclose(solver.converged, True))
        self.assertTrue(np.allclose(solver.check_convergence, True))

    @parameterized.expand(
        [(True, False, False), (False, True, False), (False, False, True)]
    )
    def test_convergence_criteria_any(self, element, qoi, estimator):
        self.parameters.convergence_criteria = "any"
        solver = self.solver(time_partition=TimePartition(1.0, 1, 0.5, []))
        solver.check_element_count_convergence = MagicMock(return_value=element)
        solver.check_qoi_convergence = MagicMock(return_value=qoi)
        solver.check_estimator_convergence = MagicMock(return_value=estimator)
        with patch("goalie.go_solver.GoalOrientedSolver.forms") as mock_forms:
            mock_forms.return_value = MagicMock()
            solver.fixed_point_iteration(
                empty_adaptor,
                parameters=self.parameters,
            )
        self.assertTrue(np.allclose(solver.check_convergence, True))
