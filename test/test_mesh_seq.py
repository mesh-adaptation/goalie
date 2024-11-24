"""
Testing for the mesh sequence objects.
"""

import re
import unittest

from firedrake import (
    Function,
    FunctionSpace,
    UnitCubeMesh,
    UnitIntervalMesh,
    UnitSquareMesh,
)
from parameterized import parameterized

from goalie.mesh_seq import MeshSeq
from goalie.time_partition import TimeInterval, TimePartition


class BaseClasses:
    """
    Base classes for mesh sequence unit testing.
    """

    class MeshSeqTestCase(unittest.TestCase):
        """
        Test case with a simple setUp method and mesh constructor.
        """

        def setUp(self):
            self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
            self.time_interval = TimeInterval(1.0, [0.5], ["field"])

        def trivial_mesh(self, dim):
            try:
                return {
                    1: UnitIntervalMesh(1),
                    2: UnitSquareMesh(1, 1),
                    3: UnitCubeMesh(1, 1, 1),
                }[dim]
            except KeyError:
                raise ValueError(f"Dimension {dim} not supported.") from None


class TestExceptions(BaseClasses.MeshSeqTestCase):
    """
    Unit tests for exceptions raised by :class:`MeshSeq`.
    """

    def test_inconsistent_dim_error(self):
        meshes = [self.trivial_mesh(2), self.trivial_mesh(3)]
        with self.assertRaises(ValueError) as cm:
            MeshSeq(self.time_partition, meshes)
        msg = "Meshes must all have the same topological dimension."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_solver"])
    def test_notimplemented_error(self, function_name):
        mesh_seq = MeshSeq(self.time_interval, self.trivial_mesh(2))
        with self.assertRaises(NotImplementedError) as cm:
            if function_name == "get_function_spaces":
                getattr(mesh_seq, function_name)(mesh_seq[0])
            else:
                getattr(mesh_seq, function_name)()
        msg = f"'{function_name}' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_return_dict_error(self, method):
        kwargs = {method: lambda _: 0}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, self.trivial_mesh(2), **kwargs)
        msg = f"{method} should return a dict"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_missing_field_error(self, method):
        kwargs = {method: lambda _: {}}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, self.trivial_mesh(2), **kwargs)
        msg = "missing fields {'field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition"])
    def test_unexpected_field_error(self, method):
        kwargs = {method: lambda _: {"field": None, "extra_field": None}}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, self.trivial_mesh(2), **kwargs)
        msg = "unexpected fields {'extra_field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    def test_solver_generator_error(self):
        mesh = self.trivial_mesh(2)
        f_space = FunctionSpace(mesh, "CG", 1)
        kwargs = {
            "get_function_spaces": lambda _: {"field": f_space},
            "get_initial_condition": lambda _: {"field": Function(f_space)},
            "get_solver": lambda _: lambda *_: {},
        }
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, mesh, **kwargs)
        msg = "solver should yield"
        self.assertEqual(str(cm.exception), msg)


class TestGeneric(BaseClasses.MeshSeqTestCase):
    """
    Generic unit tests for :class:`MeshSeq`.
    """

    def setUp(self):
        self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])

    def test_setitem(self):
        mesh1 = UnitSquareMesh(1, 1, diagonal="left")
        mesh2 = UnitSquareMesh(1, 1, diagonal="right")
        mesh_seq = MeshSeq(self.time_interval, [mesh1])
        self.assertEqual(mesh_seq[0], mesh1)
        mesh_seq[0] = mesh2
        self.assertEqual(mesh_seq[0], mesh2)

    def test_counting_2d(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [18])
        self.assertEqual(mesh_seq.count_vertices(), [16])

    def test_counting_3d(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitCubeMesh(3, 3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [162])
        self.assertEqual(mesh_seq.count_vertices(), [64])


class TestStringFormatting(BaseClasses.MeshSeqTestCase):
    """
    Test that the :meth:`__str__` and :meth:`__repr__` methods work as intended for
    Goalie's :class:`MeshSeq` object.
    """

    def test_mesh_seq_time_interval_str(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>']")

    def test_mesh_seq_time_partition_str(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>', '<Mesh ?>']")

    def test_mesh_seq_time_interval_repr(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        expected = "MeshSeq([Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_time_partition_2_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_time_partition_3_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
            UnitSquareMesh(1, 1, diagonal="left"),
        ]
        mesh_seq = MeshSeq(self.time_partition, meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "..."
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))
