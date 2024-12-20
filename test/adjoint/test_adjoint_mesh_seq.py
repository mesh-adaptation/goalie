"""
Testing for the mesh sequence objects.
"""

import logging
import unittest
from unittest.mock import patch

import pyadjoint
import pytest
from animate.utility import norm
from firedrake import (
    Function,
    FunctionSpace,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    UnitTriangleMesh,
    VectorFunctionSpace,
    dx,
    inner,
    solve,
)
from parameterized import parameterized
from pyadjoint.block_variable import BlockVariable

from goalie.go_mesh_seq import GoalOrientedMeshSeq
from goalie.log import *
from goalie.time_partition import TimeInterval, TimePartition
from goalie_adjoint import *


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
        self.mesh_seq = AdjointMeshSeq(
            self.time_interval,
            self.mesh,
            get_function_spaces=self.get_p0_spaces,
            qoi_type="end_time",
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
        mesh_seq = AdjointMeshSeq(
            time_interval,
            self.mesh,
            get_function_spaces=self.get_p0_spaces,
            qoi_type="end_time",
        )
        solve_block = MockSolveBlock()
        self.assertIsNone(mesh_seq._dependency("field", 0, solve_block))


class TestGetSolveBlocks(unittest.TestCase):
    """
    Unit tests for :meth:`get_solve_blocks`.
    """

    @staticmethod
    def get_function_spaces(mesh):
        return {"field": FunctionSpace(mesh, "R", 0)}

    def setUp(self):
        time_interval = TimeInterval(1.0, [1.0], ["field"])
        self.mesh_seq = AdjointMeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
            qoi_type="steady",
        )
        if not pyadjoint.annotate_tape():
            pyadjoint.continue_annotation()

    def tearDown(self):
        if pyadjoint.annotate_tape():
            pyadjoint.pause_annotation()

    @staticmethod
    def arbitrary_solve(sol):
        fs = sol.function_space()
        test = TestFunction(fs)
        trial = TrialFunction(fs)
        solve(test * trial * dx == test * dx, sol, ad_block_tag=sol.name())

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test_no_blocks(self):
        with self._caplog.at_level(logging.WARNING):
            blocks = self.mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = "Tape has no blocks!"
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_no_solve_blocks(self):
        fs = self.mesh_seq.function_spaces["field"][0]
        Function(fs).assign(1.0)
        with self._caplog.at_level(WARNING):
            blocks = self.mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = "Tape has no solve blocks!"
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_wrong_solve_block(self):
        fs = self.mesh_seq.function_spaces["field"][0]
        u = Function(fs, name="u")
        self.arbitrary_solve(u)
        with self._caplog.at_level(WARNING):
            blocks = self.mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = (
            "No solve blocks associated with field 'field'."
            " Has ad_block_tag been used correctly?"
        )
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_wrong_function_space(self):
        fs = FunctionSpace(self.mesh_seq[0], "CG", 1)
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        msg = (
            "Solve block list for field 'field' contains mismatching elements:"
            " <R0 on a triangle> vs. <CG1 on a triangle>."
        )
        with self.assertRaises(ValueError) as cm:
            self.mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)

    def test_too_many_timesteps(self):
        time_interval = TimeInterval(1.0, [0.5], ["field"])
        mesh_seq = AdjointMeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
            qoi_type="end_time",
        )
        fs = mesh_seq.function_spaces["field"][0]
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        msg = (
            "Number of timesteps exceeds number of solve blocks for field 'field' on"
            " subinterval 0: 2 > 1."
        )
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)

    def test_incompatible_timesteps(self):
        time_interval = TimeInterval(1.0, [0.5], ["field"])
        mesh_seq = AdjointMeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
            qoi_type="end_time",
        )
        fs = mesh_seq.function_spaces["field"][0]
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        self.arbitrary_solve(u)
        self.arbitrary_solve(u)
        msg = (
            "Number of timesteps is not divisible by number of solve blocks for field"
            " 'field' on subinterval 0: 2 vs. 3."
        )
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)


class TrivialGoalOrientedBaseClass(unittest.TestCase):
    """
    Base class for tests with a trivial :class:`GoalOrientedMeshSeq`.
    """

    def setUp(self):
        self.field = "field"
        self.time_interval = TimeInterval(1.0, [1.0], [self.field])
        self.meshes = [UnitSquareMesh(1, 1)]

    @staticmethod
    def constant_qoi(mesh_seq, solutions, index):
        R = FunctionSpace(mesh_seq[index], "R", 0)
        return lambda: Function(R).assign(1) * dx

    def go_mesh_seq(self, get_function_spaces, parameters=None):
        return GoalOrientedMeshSeq(
            self.time_interval,
            self.meshes,
            get_function_spaces=get_function_spaces,
            qoi_type="steady",
            parameters=parameters,
        )


class TestGoalOrientedMeshSeq(TrivialGoalOrientedBaseClass):
    """
    Unit tests for a :class:`GoalOrientedMeshSeq`.
    """

    def get_function_spaces(self, mesh):
        return {self.field: FunctionSpace(mesh, "R", 0)}

    def test_read_forms_error_field(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces)
        with self.assertRaises(ValueError) as cm:
            mesh_seq.read_forms({"field2": None})
        msg = (
            "Unexpected field 'field2' in forms dictionary."
            f" Expected one of ['{self.field}']."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_read_forms_error_form(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces)
        with self.assertRaises(TypeError) as cm:
            mesh_seq.read_forms({self.field: None})
        msg = f"Expected a UFL form for field '{self.field}', not '<class 'NoneType'>'."
        self.assertEqual(str(cm.exception), msg)


class TestGlobalEnrichment(TrivialGoalOrientedBaseClass):
    """
    Unit tests for global enrichment of a :class:`GoalOrientedMeshSeq`.
    """

    def get_function_spaces_decorator(self, degree, family, rank):
        def get_function_spaces(mesh):
            if rank == 0:
                return {self.field: FunctionSpace(mesh, degree, family)}
            elif rank == 1:
                return {self.field: VectorFunctionSpace(mesh, degree, family)}
            else:
                raise NotImplementedError

        return get_function_spaces

    def test_enrichment_error(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("R", 0, 0))
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_enriched_mesh_seq(enrichment_method="q")
        self.assertEqual(str(cm.exception), "Enrichment method 'q' not supported.")

    def test_num_enrichments_error(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("R", 0, 0))
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_enriched_mesh_seq(num_enrichments=0)
        msg = "A positive number of enrichments is required."
        self.assertEqual(str(cm.exception), msg)

    def test_form_error(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("R", 0, 0))
        with self.assertRaises(AttributeError) as cm:
            mesh_seq.forms()
        msg = (
            "Forms have not been read in. Use read_forms({'field_name': F}) in"
            " get_solver to read in the forms."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_h_enrichment_error(self):
        end_time = 1.0
        num_subintervals = 2
        dt = end_time / num_subintervals
        mesh_seq = GoalOrientedMeshSeq(
            TimePartition(end_time, num_subintervals, dt, "field"),
            [UnitTriangleMesh()] * num_subintervals,
            get_qoi=self.constant_qoi,
            qoi_type="end_time",
        )
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_enriched_mesh_seq(enrichment_method="h")
        msg = "h-enrichment is not supported for shallow-copied meshes."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([[1], [2]])
    def test_h_enrichment_mesh(self, num_enrichments):
        """
        Base mesh:   1 enrichment:  2 enrichments:

         o-------o     o---o---o      o-o-o-o-o
         |      /|     |  /|  /|      |/|/|/|/|
         |     / |     | / | / |      o-o-o-o-o
         |    /  |     |/  |/  |      |/|/|/|/|
         |   /   |     o---o---o      o-o-o-o-o
         |  /    |     |  /|  /|      |/|/|/|/|
         | /     |     | / | / |      o-o-o-o-o
         |/      |     |/  |/  |      |/|/|/|/|
         o-------o     o---o---o      o-o-o-o-o
        """
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("R", 0, 0))
        mesh_seq_e = mesh_seq.get_enriched_mesh_seq(
            enrichment_method="h", num_enrichments=num_enrichments
        )
        self.assertEqual(mesh_seq[0].num_cells(), 2)
        self.assertEqual(mesh_seq[0].num_vertices(), 4)
        self.assertEqual(mesh_seq[0].num_edges(), 5)
        n = num_enrichments
        self.assertEqual(mesh_seq_e[0].num_cells(), 2 * 4**n)
        self.assertEqual(mesh_seq_e[0].num_vertices(), (2 * n + 1) ** 2)
        self.assertEqual(
            mesh_seq_e[0].num_edges(),
            (2**n + 1) * (2 ** (n + 1)) + (2 ** (2 * n)),
        )

    @parameterized.expand(
        [
            ("DG", 0, 0),
            ("DG", 0, 1),
            ("CG", 1, 0),
            ("CG", 1, 1),
            ("CG", 2, 0),
            ("CG", 2, 1),
        ]
    )
    def test_h_enrichment_space(self, family, degree, rank):
        mesh_seq = self.go_mesh_seq(
            self.get_function_spaces_decorator(family, degree, rank)
        )
        mesh_seq_e = mesh_seq.get_enriched_mesh_seq(
            enrichment_method="h", num_enrichments=1
        )
        fspace = mesh_seq.function_spaces[self.field][0]
        element = fspace.ufl_element()
        enriched_fspace = mesh_seq_e.function_spaces[self.field][0]
        enriched_element = enriched_fspace.ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree(), enriched_element.degree())
        self.assertEqual(fspace.value_shape, enriched_fspace.value_shape)

    def test_p_enrichment_mesh(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("CG", 1, 0))
        mesh_seq_e = mesh_seq.get_enriched_mesh_seq(
            enrichment_method="p", num_enrichments=1
        )
        self.assertEqual(self.meshes[0], mesh_seq[0])
        self.assertEqual(self.meshes[0], mesh_seq_e[0])

    @parameterized.expand(
        [
            ("DG", 0, 0, 1),
            ("DG", 0, 0, 2),
            ("DG", 0, 1, 1),
            ("DG", 0, 1, 2),
            ("CG", 1, 0, 1),
            ("CG", 1, 0, 2),
            ("CG", 1, 1, 1),
            ("CG", 1, 1, 2),
            ("CG", 2, 0, 1),
            ("CG", 2, 0, 2),
            ("CG", 2, 1, 1),
            ("CG", 2, 1, 2),
        ]
    )
    def test_p_enrichment_space(self, family, degree, rank, num_enrichments):
        mesh_seq = self.go_mesh_seq(
            self.get_function_spaces_decorator(family, degree, rank)
        )
        mesh_seq_e = mesh_seq.get_enriched_mesh_seq(
            enrichment_method="p", num_enrichments=num_enrichments
        )
        fspace = mesh_seq.function_spaces[self.field][0]
        element = fspace.ufl_element()
        enriched_fspace = mesh_seq_e.function_spaces[self.field][0]
        enriched_element = enriched_fspace.ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree() + num_enrichments, enriched_element.degree())
        self.assertEqual(fspace.value_shape, enriched_fspace.value_shape)

    @parameterized.expand(
        [
            ("DG", 0, 0, "h", 1),
            ("DG", 0, 0, "h", 2),
            ("CG", 1, 0, "h", 1),
            ("CG", 1, 0, "h", 2),
            ("CG", 2, 0, "h", 1),
            ("CG", 2, 0, "h", 2),
            ("DG", 0, 0, "p", 1),
            ("DG", 0, 0, "p", 2),
            ("CG", 1, 0, "p", 1),
            ("CG", 1, 0, "p", 2),
            ("CG", 2, 0, "p", 1),
            ("CG", 2, 0, "p", 2),
            ("DG", 0, 1, "h", 1),
            ("DG", 0, 1, "h", 2),
            ("CG", 1, 1, "h", 1),
            ("CG", 1, 1, "h", 2),
            ("CG", 2, 1, "h", 1),
            ("CG", 2, 1, "h", 2),
            ("DG", 0, 1, "p", 1),
            ("DG", 0, 1, "p", 2),
            ("CG", 1, 1, "p", 1),
            ("CG", 1, 1, "p", 2),
            ("CG", 2, 1, "p", 1),
            ("CG", 2, 1, "p", 2),
        ]
    )
    def test_enrichment_transfer(
        self, family, degree, rank, enrichment_method, num_enrichments
    ):
        mesh_seq = self.go_mesh_seq(
            self.get_function_spaces_decorator(family, degree, rank)
        )
        mesh_seq_e = mesh_seq.get_enriched_mesh_seq(
            enrichment_method=enrichment_method, num_enrichments=num_enrichments
        )
        transfer = mesh_seq._get_transfer_function(enrichment_method)
        source = Function(mesh_seq.function_spaces["field"][0])
        x = SpatialCoordinate(mesh_seq[0])
        source.project(x if rank == 1 else sum(x))
        target = Function(mesh_seq_e.function_spaces["field"][0])
        transfer(source, target)
        self.assertAlmostEqual(norm(source), norm(target))


class GoalOrientedBaseClass(unittest.TestCase):
    """
    Base class for tests with a complete :class:`GoalOrientedMeshSeq`.
    """

    def setUp(self):
        self.field = "field"
        self.time_partition = TimePartition(1.0, 1, 0.5, [self.field])
        self.meshes = [UnitSquareMesh(1, 1)]

    def go_mesh_seq(self, coeff_diff=0.0):
        def get_function_spaces(mesh):
            return {self.field: FunctionSpace(mesh, "R", 0)}

        def get_initial_condition(mesh_seq):
            return {self.field: Function(mesh_seq.function_spaces[self.field][0])}

        def get_solver(mesh_seq):
            def solver(index):
                tp = mesh_seq.time_partition
                R = FunctionSpace(mesh_seq[index], "R", 0)
                dt = Function(R).assign(tp.timesteps[index])

                u, u_ = mesh_seq.fields[self.field]
                f = Function(R).assign(1.0001)
                v = TestFunction(u.function_space())
                F = (u - u_) / dt * v * dx - f * v * dx
                mesh_seq.read_forms({self.field: F})

                for _ in range(tp.num_timesteps_per_subinterval[index]):
                    solve(F == 0, u, ad_block_tag=self.field)
                    yield

                    u_.assign(u)
                    f += coeff_diff

            return solver

        def get_qoi(mesh_seq, i):
            def end_time_qoi():
                u = mesh_seq.fields[self.field][0]
                return inner(u, u) * dx

            return end_time_qoi

        return GoalOrientedMeshSeq(
            self.time_partition,
            self.meshes,
            get_initial_condition=get_initial_condition,
            get_function_spaces=get_function_spaces,
            get_solver=get_solver,
            get_qoi=get_qoi,
            qoi_type="end_time",
        )


class TestDetectChangedCoefficients(GoalOrientedBaseClass):
    """
    Unit tests for detecting changed coefficients using
    :meth:`GoalOrientedMeshSeq._detect_changing_coefficients`.
    """

    def test_constant_coefficients(self):
        mesh_seq = self.go_mesh_seq()
        adj_gen = mesh_seq._solve_adjoint(track_coefficients=True)
        # Solve over the first (only) subinterval
        next(adj_gen)
        # No coefficients have changed
        self.assertEqual(mesh_seq._changed_form_coeffs, {self.field: {}})

    def test_changed_coefficients(self):
        # Change coefficient f by coeff_diff every timestep
        coeff_diff = 1.1
        mesh_seq = self.go_mesh_seq(coeff_diff=coeff_diff)
        adj_gen = mesh_seq._solve_adjoint(track_coefficients=True)
        # Solve over the first (only) subinterval
        next(adj_gen)
        changed_coeffs_dict = mesh_seq._changed_form_coeffs[self.field]
        coeff_idx = next(iter(changed_coeffs_dict))
        for export_idx, f in changed_coeffs_dict[coeff_idx].items():
            self.assertTrue(f.vector().gather() == [1.0001 + export_idx * coeff_diff])
