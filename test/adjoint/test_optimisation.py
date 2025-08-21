"""
Unit tests for the optimisation module.
"""

import unittest

import ufl
from firedrake.exceptions import ConvergenceError
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import VertexOnlyMesh
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitIntervalMesh

from goalie.adjoint import AdjointMeshSeq, annotate_qoi
from goalie.field import Field
from goalie.optimisation import OptimisationProgress, QoIOptimiser
from goalie.options import OptimisationParameters
from goalie.time_partition import TimeInstant, TimeInterval


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

    def test_convert_for_output(self):
        mesh = VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])
        R = FunctionSpace(mesh, "R", 0)
        self.progress["qoi"] = [Function(R).assign(1.0)]
        self.progress["control"] = [Function(R).assign(2.0)]
        self.progress["gradient"] = [Function(R).assign(3.0)]
        self.progress.convert_for_output()
        self.assertEqual(self.progress["qoi"], [1.0])
        self.assertEqual(self.progress["control"], [2.0])
        self.assertEqual(self.progress["gradient"], [3.0])


class TestQoIOptimiserExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised by the :class:`goalie.optimisation.QoIOpimiser`
    class.
    """

    def setUp(self):
        self.parameters = OptimisationParameters()
        self.mesh = VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])

    def time_partition(self, family="Real", degree=0):
        field = Field("field", family=family, degree=degree)
        return TimeInterval(1, 1.0, field)

    def test_invalid_control_error(self):
        mesh_seq = AdjointMeshSeq(self.time_partition(), self.mesh, qoi_type="steady")
        with self.assertRaises(ValueError) as cm:
            _ = QoIOptimiser(mesh_seq, "control", self.parameters)
        self.assertEqual(str(cm.exception), "Invalid control 'control'.")

    def test_not_r_space_error(self):
        time_partition = self.time_partition(family="CG", degree=1)
        mesh = UnitIntervalMesh(1)
        mesh_seq = AdjointMeshSeq(time_partition, mesh, qoi_type="steady")
        with self.assertRaises(NotImplementedError) as cm:
            _ = QoIOptimiser(mesh_seq, "field", self.parameters)
        msg = "Only controls in R-space are currently implemented."
        self.assertEqual(str(cm.exception), msg)

    def test_divergence_error(self):
        mesh_seq = AdjointMeshSeq(self.time_partition(), self.mesh, qoi_type="steady")
        optimiser = QoIOptimiser(mesh_seq, "field", self.parameters)
        optimiser.progress["qoi"] = [1.0]
        mesh_seq.J = 2.0
        with self.assertRaises(ConvergenceError) as cm:
            optimiser.check_qoi_divergence()
        self.assertEqual(str(cm.exception), "QoI divergence detected.")

    def test_maxiter_error(self):
        parameters = OptimisationParameters({"maxiter": 0})
        mesh_seq = AdjointMeshSeq(self.time_partition(), self.mesh, qoi_type="steady")
        optimiser = QoIOptimiser(mesh_seq, "field", parameters)
        with self.assertRaises(ConvergenceError) as cm:
            optimiser.minimise()
        self.assertEqual(str(cm.exception), "Reached maximum number of iterations.")

    def test_method_key_error(self):
        mesh_seq = AdjointMeshSeq(self.time_partition(), self.mesh, qoi_type="steady")
        with self.assertRaises(ValueError) as cm:
            _ = QoIOptimiser(mesh_seq, "field", self.parameters, method="method")
        self.assertEqual(str(cm.exception), "Method 'method' not supported.")


class SimpleMeshSeq(AdjointMeshSeq):
    """
    Simple :class:`goalie.adjoint.AdjointMeshSeq` for optimising the scalar equation

    .. math::
        a u = b

    for scalar :math:`a` with a given :math:`b`, where :math:`u` is the prognostic
    solution. The QoI is defined such that the optimal value of :math:`u` is unity,
    which is achieved with :math:`a = 2`.
    """

    def get_initial_condition(self):
        return {
            "a": Function(self.function_spaces["a"][0]).assign(1.0),
            "u": Function(self.function_spaces["u"][0]),
        }

    def get_solver(self):
        def solver(index):
            u = self.field_functions["u"]
            a = self.field_functions["a"]
            R = self.function_spaces["u"][index]
            b = Function(R).assign(2.0)
            F = (a * u - b) * TestFunction(R) * ufl.dx
            sp = {"ksp_type": "preonly", "pc_type": "jacobi"}
            solve(F == 0, u, solver_parameters=sp, ad_block_tag="u")
            yield

        return solver

    @annotate_qoi
    def get_qoi(self, index):
        R = self.function_spaces["u"][index]

        def steady_qoi():
            sol = Function(R).assign(1.0)
            u = self.field_functions["u"]
            return ufl.inner(u - sol, u - sol) * ufl.dx

        return steady_qoi


class TestQoIOptimiserConvergence(unittest.TestCase):
    """
    Unit tests for convergence of the :class:`goalie.optimisation.QoIOptimiser` class.
    """

    def setUp(self):
        self.parameters = OptimisationParameters({"lr": 0.1, "maxiter": 10})
        mesh = VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])
        fields = [
            Field("u", family="Real", degree=0, unsteady=False),
            Field("a", family="Real", degree=0, unsteady=False, solved_for=False),
        ]
        self.mesh_seq = SimpleMeshSeq(TimeInstant(fields), mesh, qoi_type="steady")

    def test_gradient_converged(self):
        optimiser = QoIOptimiser(self.mesh_seq, "a", self.parameters)
        optimiser.progress["gradient"] = [1.0]
        self.mesh_seq._gradient = {"a": 1e-06}
        self.assertTrue(optimiser.check_gradient_convergence())

    def test_gradient_not_converged(self):
        optimiser = QoIOptimiser(self.mesh_seq, "a", self.parameters)
        optimiser.progress["gradient"] = [1.0]
        self.mesh_seq._gradient = {"a": 1.0}
        self.assertFalse(optimiser.check_gradient_convergence())

    def test_minimise(self):
        optimiser = QoIOptimiser(self.mesh_seq, "a", self.parameters)
        optimiser.minimise()
        self.assertTrue(optimiser.check_gradient_convergence())


if __name__ == "__main__":
    unittest.main()
