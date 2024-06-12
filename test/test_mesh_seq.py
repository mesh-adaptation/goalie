"""
Testing for the mesh sequence objects.
"""

import re
import unittest

from firedrake import Function, FunctionSpace, UnitCubeMesh, UnitSquareMesh
from parameterized import parameterized

from goalie.mesh_seq import MeshSeq
from goalie.time_partition import TimeInterval, TimePartition


class TestGeneric(unittest.TestCase):
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

    def test_inconsistent_dim(self):
        meshes = [UnitSquareMesh(1, 1), UnitCubeMesh(1, 1, 1)]
        with self.assertRaises(ValueError) as cm:
            MeshSeq(self.time_partition, meshes)
        msg = "Meshes must all have the same topological dimension."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_form", "get_solver"])
    def test_notimplemented_error(self, function_name):
        mesh_seq = MeshSeq(self.time_interval, UnitSquareMesh(1, 1))
        with self.assertRaises(NotImplementedError) as cm:
            if function_name == "get_function_spaces":
                getattr(mesh_seq, function_name)(mesh_seq[0])
            else:
                getattr(mesh_seq, function_name)()
        msg = f"'{function_name}' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition", "get_form"])
    def test_return_dict_error(self, method):
        mesh = UnitSquareMesh(1, 1)
        methods = ["get_function_spaces", "get_initial_condition", "get_form"]
        funcs = [lambda _: 0, lambda _: 0, lambda _: lambda *_: 0]
        methods_map = dict(zip(methods, funcs))
        if method == "get_form":
            kwargs = {method: func for method, func in methods_map.items()}
            f_space = FunctionSpace(mesh, "CG", 1)
            kwargs["get_function_spaces"] = lambda _: {"field": f_space}
            kwargs["get_initial_condition"] = lambda _: {"field": Function(f_space)}
        else:
            kwargs = {method: methods_map[method]}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, mesh, **kwargs)
        msg = f"{method} should return a dict"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition", "get_form"])
    def test_missing_field_error(self, method):
        mesh = UnitSquareMesh(1, 1)
        methods = ["get_function_spaces", "get_initial_condition", "get_form"]
        funcs = [lambda _: {}, lambda _: {}, lambda _: lambda *_: {}]
        kwargs = dict(zip(methods, funcs))
        if method == "get_form":
            f_space = FunctionSpace(mesh, "CG", 1)
            kwargs["get_function_spaces"] = lambda _: {"field": f_space}
            kwargs["get_initial_condition"] = lambda _: {"field": Function(f_space)}
        else:
            kwargs = {method: kwargs[method]}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, mesh, **kwargs)
        msg = "missing fields {'field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["get_function_spaces", "get_initial_condition", "get_form"])
    def test_unexpected_field_error(self, method):
        mesh = UnitSquareMesh(1, 1)
        methods = ["get_function_spaces", "get_initial_condition", "get_form"]
        out_dict = {"field": None, "extra_field": None}
        funcs = [lambda _: out_dict, lambda _: out_dict, lambda _: lambda *_: out_dict]
        kwargs = dict(zip(methods, funcs))
        if method == "get_form":
            f_space = FunctionSpace(mesh, "CG", 1)
            kwargs["get_function_spaces"] = lambda _: {"field": f_space}
            kwargs["get_initial_condition"] = lambda _: {"field": Function(f_space)}
        else:
            kwargs = {method: kwargs[method]}
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, mesh, **kwargs)
        msg = "unexpected fields {'extra_field'} in " + f"{method}"
        self.assertEqual(str(cm.exception), msg)

    def test_solver_generator_error(self):
        mesh = UnitSquareMesh(1, 1)
        f_space = FunctionSpace(mesh, "CG", 1)
        kwargs = {
            "get_function_spaces": lambda _: {"field": f_space},
            "get_initial_condition": lambda _: {"field": Function(f_space)},
            "get_form": lambda msq: lambda *_: {"field": msq.fields["field"][0]},
            "get_solver": lambda _: lambda *_: {},
        }
        with self.assertRaises(AssertionError) as cm:
            MeshSeq(self.time_interval, mesh, **kwargs)
        msg = "solver should yield"
        self.assertEqual(str(cm.exception), msg)

    def test_counting_2d(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [18])
        self.assertEqual(mesh_seq.count_vertices(), [16])

    def test_counting_3d(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitCubeMesh(3, 3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [162])
        self.assertEqual(mesh_seq.count_vertices(), [64])


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__` and :meth:`__repr__` methods work as intended for
    Goalie's :class:`MeshSeq` object.
    """

    def setUp(self):
        self.time_partition = TimePartition(1.0, 2, [0.5, 0.5], ["field"])
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])

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
