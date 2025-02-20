"""
Testing for gradient computation.
"""

import unittest

import ufl
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mesh import VertexOnlyMesh
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction
from firedrake.utility_meshes import UnitIntervalMesh

from goalie.adjoint import AdjointMeshSeq
from goalie.time_partition import TimeInterval, TimePartition


def point_mesh():
    return VertexOnlyMesh(UnitIntervalMesh(1), [[0.5]])


class TestExceptions(unittest.TestCase):
    """
    Unit tests for exceptions raised during gradient computation.
    """

    def test_attribute_error(self):
        mesh_seq = AdjointMeshSeq(
            TimeInterval(1.0, 1.0, "field"),
            point_mesh(),
            qoi_type="steady",
        )
        with self.assertRaises(AttributeError) as cm:
            _ = mesh_seq.gradient
        msg = (
            "To compute the gradient, pass compute_gradient=True to the solve_adjoint"
            " method."
        )
        self.assertEqual(str(cm.exception), msg)


class TestSingleSubinterval(unittest.TestCase):
    """
    Unit tests for gradient computation on mesh sequences with a single subinterval.
    """

    def setUp(self):
        self.field = "field"

    def time_partition(self, dt):
        return TimeInterval(1.0, dt, self.field)

    def test_single_timestep(self):
        time_partition = self.time_partition(1.0)

        def get_function_spaces(mesh):
            return {self.field: FunctionSpace(mesh, "R", 0)}

        def get_initial_condition(mesh_seq):
            u = Function(mesh_seq.function_spaces[self.field][0])
            u.assign(1.0)
            return {self.field: u}

        def get_solver(mesh_seq):
            def solver(index):
                """
                Artificial solve that just assigns the initial condition.
                """
                R = mesh_seq.function_spaces[self.field][index]
                u = mesh_seq.fields[self.field]
                u0 = Function(R).assign(u)
                v = TestFunction(R)
                F = u * v * ufl.dx - u0 * v * ufl.dx
                solve(F == 0, u, ad_block_tag=self.field)
                yield

            return solver

        def get_qoi(mesh_seq, index):
            # TODO: Test some more interesting QoIs with parameterized
            def steady_qoi():
                """
                QoI that squares the solution field.
                """
                u = mesh_seq.fields[self.field]
                return ufl.inner(u, u) * ufl.dx

            return steady_qoi

        mesh_seq = AdjointMeshSeq(
            time_partition,
            point_mesh(),
            get_initial_condition=get_initial_condition,
            get_function_spaces=get_function_spaces,
            get_solver=get_solver,
            get_qoi=get_qoi,
            qoi_type="steady",
        )
        mesh_seq.solve_adjoint(compute_gradient=True)
        self.assertAlmostEqual(float(mesh_seq.gradient[0]), 2.0)

    def test_two_timesteps(self):
        raise NotImplementedError("TODO")  # TODO


class TestTwoSubintervals(unittest.TestCase):
    """
    Unit tests for gradient computation on mesh sequences with two subintervals.
    """

    def setUp(self):
        end_time = 1.0
        num_subintervals = 2
        dt = end_time / num_subintervals
        self.time_partition = TimePartition(end_time, num_subintervals, dt, "field")
        self.meshes = [point_mesh() for _ in range(num_subintervals)]

    def test_single_timestep(self):
        raise NotImplementedError("TODO")  # TODO

    def test_two_timesteps(self):
        raise NotImplementedError("TODO")  # TODO
