from firedrake import *
from goalie_adjoint import *
from setup_adjoint_tests import *
import abc
from parameterized import parameterized
import unittest


def get_qoi(mesh_seq, solutions, index):
    R = FunctionSpace(mesh_seq[index], "R", 0)

    def qoi():
        return Function(R).assign(1 if mesh_seq.fp_iteration % 2 == 0 else 2) * dx

    return qoi


class MeshSeqBaseClass:
    """
    Base class for :meth:`fixed_point_iteration` unit tests.
    """

    @abc.abstractmethod
    def setUp(self):
        pass

    def test_update_params(self):
        def update_params(params, fp_iteration):
            params.element_rtol = fp_iteration

        self.parameters.miniter = self.parameters.maxiter
        self.parameters.element_rtol = 0.5
        mesh_seq = self.mesh_seq(TimeInstant([]), UnitTriangleMesh())
        mesh_seq.fixed_point_iteration(empty_adaptor, update_params=update_params)
        self.assertEqual(self.parameters.element_rtol + 1, 5)


class TestMeshSeq(unittest.TestCase, MeshSeqBaseClass):
    """
    Unit tests for :meth:`MeshSeq.fixed_point_iteration`.
    """

    def setUp(self):
        self.parameters = AdaptParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def mesh_seq(self, time_partition, mesh, parameters=None):
        return MeshSeq(
            time_partition,
            mesh,
            get_function_spaces=empty_get_function_spaces,
            get_form=empty_get_form,
            get_bcs=empty_get_bcs,
            get_solver=empty_get_solver,
            parameters=parameters or self.parameters,
        )

    def test_convergence_noop(self):
        miniter = self.parameters.miniter
        mesh_seq = self.mesh_seq(TimeInstant([]), UnitTriangleMesh())
        mesh_seq.fixed_point_iteration(empty_adaptor)
        self.assertEqual(len(mesh_seq.element_counts), miniter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_noconvergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False]

        maxiter = self.parameters.maxiter
        mesh_seq = self.mesh_seq(TimeInstant([]), mesh2)
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), maxiter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    @parameterized.expand([[True], [False]])
    def test_dropout(self, drop_out_converged):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        ap = AdaptParameters(self.parameters)
        ap.update({"drop_out_converged": drop_out_converged})
        mesh_seq = self.mesh_seq(time_partition, mesh2, parameters=ap)

        def adaptor(mesh_seq, sols):
            mesh_seq[1] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [True, False]))
        self.assertTrue(
            np.allclose(mesh_seq.check_convergence, [not drop_out_converged, True])
        )

    def test_no_late_convergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        ap = AdaptParameters(self.parameters)
        ap.update({"drop_out_converged": True})
        mesh_seq = self.mesh_seq(time_partition, mesh2, parameters=ap)

        def adaptor(mesh_seq, sols):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [False, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [True, True]))


class TestGoalOrientedMeshSeq(unittest.TestCase, MeshSeqBaseClass):
    """
    Unit tests for :meth:`GoalOrientedMeshSeq.fixed_point_iteration`.
    """

    def setUp(self):
        self.parameters = GoalOrientedParameters(
            {
                "miniter": 3,
                "maxiter": 5,
            }
        )

    def mesh_seq(self, time_partition, mesh, parameters=None, qoi_type="steady"):
        return GoalOrientedMeshSeq(
            time_partition,
            mesh,
            get_function_spaces=empty_get_function_spaces,
            get_form=empty_get_form,
            get_bcs=empty_get_bcs,
            get_solver=empty_get_solver,
            get_qoi=get_qoi,
            parameters=parameters or self.parameters,
            qoi_type=qoi_type,
        )

    def test_convergence_noop(self):
        miniter = self.parameters.miniter
        mesh_seq = self.mesh_seq(TimeInstant([]), UnitTriangleMesh())
        mesh_seq.fixed_point_iteration(empty_adaptor)
        self.assertEqual(len(mesh_seq.element_counts), miniter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, True))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    def test_noconvergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False]

        maxiter = self.parameters.maxiter
        mesh_seq = self.mesh_seq(TimeInstant([]), mesh2)
        mesh_seq.fixed_point_iteration(adaptor)
        self.assertEqual(len(mesh_seq.element_counts), maxiter + 1)
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))

    @parameterized.expand([[True], [False]])
    def test_dropout(self, drop_out_converged):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        ap = GoalOrientedParameters(self.parameters)
        ap.update({"drop_out_converged": drop_out_converged})
        mesh_seq = self.mesh_seq(
            time_partition, mesh2, parameters=ap, qoi_type="end_time"
        )

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[1] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [1, 2], [1, 1], [1, 2], [1, 1], [1, 2]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [True, False]))
        self.assertTrue(
            np.allclose(mesh_seq.check_convergence, [not drop_out_converged, True])
        )

    def test_no_late_convergence(self):
        mesh1 = UnitSquareMesh(1, 1)
        mesh2 = UnitTriangleMesh()
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [])
        ap = GoalOrientedParameters(self.parameters)
        ap.update({"drop_out_converged": True})
        mesh_seq = self.mesh_seq(
            time_partition, mesh2, parameters=ap, qoi_type="end_time"
        )

        def adaptor(mesh_seq, sols, indicators):
            mesh_seq[0] = mesh1 if mesh_seq.fp_iteration % 2 == 0 else mesh2
            return [False, False]

        mesh_seq.fixed_point_iteration(adaptor)
        expected = [[1, 1], [2, 1], [1, 1], [2, 1], [1, 1], [2, 1]]
        self.assertEqual(mesh_seq.element_counts, expected)
        self.assertTrue(np.allclose(mesh_seq.converged, [False, False]))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, [True, True]))

    def test_convergence_criteria_all(self):
        mesh = UnitSquareMesh(1, 1)
        time_partition = TimePartition(1.0, 1, 0.5, [])
        ap = GoalOrientedParameters(self.parameters)
        ap.update({"convergence_criteria": "all"})
        mesh_seq = self.mesh_seq(
            time_partition, mesh, parameters=ap, qoi_type="end_time"
        )
        mesh_seq.fixed_point_iteration(empty_adaptor)
        self.assertTrue(np.allclose(mesh_seq.element_counts, 2))
        self.assertTrue(np.allclose(mesh_seq.converged, False))
        self.assertTrue(np.allclose(mesh_seq.check_convergence, True))
