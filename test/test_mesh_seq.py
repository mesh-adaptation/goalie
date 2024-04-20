"""
Testing for the mesh sequence objects.
"""

import re
import unittest
from unittest.mock import patch

from firedrake import (
    Function,
    FunctionSpace,
    UnitCubeMesh,
    UnitSquareMesh,
    UnitTriangleMesh,
)
from parameterized import parameterized
from pyadjoint.block_variable import BlockVariable

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
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(1, 1)])
        with self.assertRaises(NotImplementedError) as cm:
            if function_name == "get_function_spaces":
                getattr(mesh_seq, function_name)(mesh_seq[0])
            else:
                getattr(mesh_seq, function_name)()
        msg = f"'{function_name}' needs implementing."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["initial condition", "form"])
    def test_missing_field_error(self, function_name):
        mesh = UnitSquareMesh(1, 1)

        if function_name == "initial condition":

            def get_initial_condition(mesh_seq):
                return {"new_field": 1.0}

            mesh_seq = MeshSeq(
                self.time_interval, mesh, get_initial_condition=get_initial_condition
            )
            with self.assertRaises(AssertionError) as cm:
                mesh_seq.initial_condition()
        elif function_name == "form":

            def get_form(mesh_seq):
                def form(index, solutions):
                    return {"new_field": 1.0}

                return form

            mesh_seq = MeshSeq(self.time_interval, mesh, get_form=get_form)
            with self.assertRaises(AssertionError) as cm:
                mesh_seq.form(0, 0)

        msg = "missing fields {'field'} in " + f"{function_name}"
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand(["initial condition", "form"])
    def test_unexpected_field_error(self, function_name):
        mesh = UnitSquareMesh(1, 1)

        if function_name == "initial condition":

            def get_initial_condition(mesh_seq):
                return {"field": 1.0, "extra_field": 1.0}

            mesh_seq = MeshSeq(
                self.time_interval, mesh, get_initial_condition=get_initial_condition
            )
            with self.assertRaises(AssertionError) as cm:
                mesh_seq.initial_condition()
        elif function_name == "form":

            def get_form(mesh_seq):
                def form(index, solutions):
                    return {"field": 1.0, "extra_field": 1.0}

                return form

            mesh_seq = MeshSeq(self.time_interval, mesh, get_form=get_form)
            with self.assertRaises(AssertionError) as cm:
                mesh_seq.form(0, 0)

        msg = "unexpected fields {'extra_field'} in " + f"{function_name}"
        self.assertEqual(str(cm.exception), msg)

    def test_counting(self):
        mesh_seq = MeshSeq(self.time_interval, [UnitSquareMesh(3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [18])
        self.assertEqual(mesh_seq.count_vertices(), [16])


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


class TestBlockLogic(unittest.TestCase):
    """
    Unit tests for :meth:`MeshSeq._dependency` and :meth:`MeshSeq._output`.
    """

    @staticmethod
    def get_p0_spaces(mesh):
        return {"field": FunctionSpace(mesh, "DG", 0)}

    def setUp(self):
        self.time_interval = TimeInterval(1.0, 0.5, "field")
        self.mesh = UnitTriangleMesh()
        self.mesh_seq = MeshSeq(
            self.time_interval, self.mesh, get_function_spaces=self.get_p0_spaces
        )

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_not_function(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(1)
        solve_block._outputs = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_wrong_function_space(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(FunctionSpace(self.mesh, "CG", 1)))
        solve_block._outputs = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_wrong_name(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field2"))
        solve_block._outputs = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_valid(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field"))
        solve_block._outputs = [block_variable]
        self.assertIsNotNone(self.mesh_seq._output("field", 0, solve_block))

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_multiple_valid_error(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field"))
        solve_block._outputs = [block_variable, block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._output("field", 0, solve_block)
        msg = (
            "Cannot determine a unique output index for the solution associated with"
            " field 'field' out of 2 candidates."
        )
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_not_function(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(1)
        solve_block._dependencies = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_wrong_function_space(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(FunctionSpace(self.mesh, "CG", 1)))
        solve_block._dependencies = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_wrong_name(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field_new"))
        solve_block._dependencies = [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_valid(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field_old"))
        solve_block._dependencies = [block_variable]
        self.assertIsNotNone(self.mesh_seq._dependency("field", 0, solve_block))

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_multiple_valid_error(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "DG", 0)
        block_variable = BlockVariable(Function(function_space, name="field_old"))
        solve_block._dependencies = [block_variable, block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.mesh_seq._dependency("field", 0, solve_block)
        msg = (
            "Cannot determine a unique dependency index for the lagged solution"
            " associated with field 'field' out of 2 candidates."
        )
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_steady(self, MockSolveBlock):
        time_interval = TimeInterval(1.0, 0.5, "field", field_types="steady")
        mesh_seq = MeshSeq(
            time_interval, self.mesh, get_function_spaces=self.get_p0_spaces
        )
        solve_block = MockSolveBlock()
        self.assertIsNone(mesh_seq._dependency("field", 0, solve_block))
