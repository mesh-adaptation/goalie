import abc
import unittest
from unittest.mock import MagicMock

from firedrake import *
from parameterized import parameterized

from goalie_adjoint import *


def constant_qoi(mesh_seq, index):
    R = FunctionSpace(mesh_seq[index], "R", 0)

    def qoi():
        return Function(R).assign(1) * dx

    return qoi


def oscillating_qoi(mesh_seq, index):
    R = FunctionSpace(mesh_seq[index], "R", 0)

    def qoi():
        return Function(R).assign(1 if mesh_seq.fp_iteration % 2 == 0 else 2) * dx

    return qoi


def oscillating_adaptor0(mesh_seq, *args):
    mesh_seq[0] = (
        UnitSquareMesh(1, 1) if mesh_seq.fp_iteration % 2 == 0 else UnitTriangleMesh()
    )
    return [False] * len(mesh_seq)


def oscillating_adaptor1(mesh_seq, *args):
    mesh_seq[1] = (
        UnitSquareMesh(1, 1) if mesh_seq.fp_iteration % 2 == 0 else UnitTriangleMesh()
    )
    return [False] * len(mesh_seq)


def empty_adaptor(*args):
    return [False]


class MeshSeqBaseClass:
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
            "mesh": UnitTriangleMesh(),
        }

    def mesh_seq(self, **kwargs):
        kw = self.default_kwargs
        kw.update(kwargs)
        mesh_seq = self.seq(kw.pop("time_partition"), kw.pop("mesh"), **kw)
        mesh_seq._get_function_spaces = lambda _: {}
        mesh_seq._get_solver = lambda _: lambda *_: (
            yield from (None for _ in iter(int, 1))
        )
        mesh_seq.params = self.parameters
        return mesh_seq

    def test_convergence_noop(self):
        miniter = self.parameters.miniter
        mesh_seq = self.mesh_seq()
        mesh_seq.fixed_point_iteration(empty_adaptor, parameters=self.parameters)
        self.assertEqual(len(mesh_seq.element_counts), miniter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_noconvergence(self):
        maxiter = self.parameters.maxiter
        mesh_seq = self.mesh_seq()
        mesh_seq.fixed_point_iteration(oscillating_adaptor0, parameters=self.parameters)
        self.assertEqual(len(mesh_seq.element_counts), maxiter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_no_late_convergence(self):
        self.parameters.drop_out_converged = True
        mesh_seq = self.mesh_seq(time_partition=TimePartition(1.0, 2, [0.5, 0.5], []))
        mesh_seq.fixed_point_iteration(oscillating_adaptor0, parameters=self.parameters)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [False, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [True, True]))

    @parameterized.expand([[True], [False]])
    def test_dropout(self, drop_out_converged):
        self.parameters.drop_out_converged = drop_out_converged
        mesh_seq = self.mesh_seq(time_partition=TimePartition(1.0, 2, [0.5, 0.5], []))
        mesh_seq.fixed_point_iteration(oscillating_adaptor1, parameters=self.parameters)
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [True, False]))
        self.assertTrue(
            np.allclose(mesh_seq.check_convergence, [not drop_out_converged, True])
        )

    def test_update_params(self):
        def update_params(params, fp_iteration):
            params.element_rtol = fp_iteration

        self.parameters.miniter = self.parameters.maxiter
        self.parameters.element_rtol = 0.5
        mesh_seq = self.mesh_seq()
        mesh_seq.fixed_point_iteration(
            empty_adaptor, parameters=self.parameters, update_params=update_params
        )
        self.assertEqual(self.parameters.element_rtol + 1, 5)

    def test_convergence_lt_miniter(self):
        mesh_seq = self.mesh_seq()
        self.assertFalse(self.check_convergence(mesh_seq))
        self.assertFalse(mesh_seq.converged)

    def test_convergence_true(self):
        self.parameters.drop_out_converged = True
        mesh_seq = self.mesh_seq()
        self.set_values(mesh_seq, np.ones((mesh_seq.params.miniter + 1, 1)))
        self.assertTrue(self.check_convergence(mesh_seq))
        return mesh_seq

    def test_convergence_false(self):
        self.parameters.drop_out_converged = True
        mesh_seq = self.mesh_seq()
        values = np.ones((self.parameters.miniter + 1, 1))
        values[-1] = 2
        self.set_values(mesh_seq, values)
        self.assertFalse(self.check_convergence(mesh_seq))
        return mesh_seq

    def test_convergence_check_false(self):
        self.parameters.drop_out_converged = True
        mesh_seq = self.mesh_seq()
        self.set_values(mesh_seq, np.ones((mesh_seq.params.miniter + 1, 1)))
        mesh_seq.check_convergence[:] = True
        mesh_seq.check_convergence[-1] = False
        self.assertFalse(self.check_convergence(mesh_seq))


class TestMeshSeq(unittest.TestCase, MeshSeqBaseClass):
    """
    Unit tests for :meth:`MeshSeq.fixed_point_iteration`.
    """

    seq = MeshSeq

    def setUp(self):
        self.parameters = AdaptParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def set_values(self, mesh_seq, value):
        mesh_seq.element_counts = value

    def check_convergence(self, mesh_seq):
        return mesh_seq.check_element_count_convergence()

    def test_converged_array_true(self):
        self.assertTrue(super().test_convergence_true().converged)

    def test_converged_array_false(self):
        self.assertFalse(super().test_convergence_false().converged)


class TestAdjointMeshSeq(unittest.TestCase, MeshSeqBaseClass):
    """
    Unit tests for :meth:`AdjointMeshSeq.fixed_point_iteration`.
    """

    seq = AdjointMeshSeq

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

    def mesh_seq(self, **kwargs):
        kw = self.default_kwargs.copy()
        kw.update(kwargs)
        tp = kw["time_partition"]
        num_timesteps = 1 if tp is None else tp.num_timesteps
        kw["qoi_type"] = "steady" if num_timesteps == 1 else "end_time"
        return super().mesh_seq(**kw)

    def set_values(self, mesh_seq, value):
        mesh_seq.qoi_values = value

    def check_convergence(self, mesh_seq):
        return mesh_seq.check_qoi_convergence()


class TestGoalOrientedMeshSeq(TestAdjointMeshSeq):
    """
    Unit tests for :meth:`GoalOrientedMeshSeq.fixed_point_iteration`.
    """

    seq = GoalOrientedMeshSeq

    def set_values(self, mesh_seq, value):
        mesh_seq.estimator_values = value

    def check_convergence(self, mesh_seq):
        return mesh_seq.check_estimator_convergence()

    def test_convergence_criteria_all_false(self):
        self.parameters.convergence_criteria = "all"
        mesh_seq = self.mesh_seq(time_partition=TimePartition(1.0, 1, 0.5, []))
        mesh_seq.fixed_point_iteration(empty_adaptor, parameters=self.parameters)
        self.assertTrue(np.allclose(mesh_seq.element_counts, 1))
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_convergence_criteria_all_true(self):
        self.parameters.convergence_criteria = "all"
        mesh_seq = self.mesh_seq(
            time_partition=TimePartition(1.0, 1, 0.5, []),
            get_qoi=constant_qoi,
        )
        mesh_seq.error_estimate = MagicMock(return_value=1)
        mesh_seq.fixed_point_iteration(empty_adaptor, parameters=self.parameters)
        self.assertTrue(np.allclose(mesh_seq.element_counts, 1))
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    @parameterized.expand(
        [(True, False, False), (False, True, False), (False, False, True)]
    )
    def test_convergence_criteria_any(self, element, qoi, estimator):
        self.parameters.convergence_criteria = "any"
        mesh_seq = self.mesh_seq(time_partition=TimePartition(1.0, 1, 0.5, []))
        mesh_seq.check_element_count_convergence = MagicMock(return_value=element)
        mesh_seq.check_qoi_convergence = MagicMock(return_value=qoi)
        mesh_seq.check_estimator_convergence = MagicMock(return_value=estimator)
        mesh_seq.fixed_point_iteration(empty_adaptor, parameters=self.parameters)
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))
