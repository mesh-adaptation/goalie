"""
Testing for the mesh sequence objects.
"""

import logging
import unittest
from unittest.mock import patch

import pyadjoint
import pytest
import ufl
from animate.utility import norm
from finat.ufl import FiniteElement, VectorElement
from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.solving import solve
from firedrake.ufl_expr import TestFunction, TrialFunction
from firedrake.utility_meshes import UnitSquareMesh, UnitTriangleMesh
from parameterized import parameterized
from pyadjoint.block_variable import BlockVariable

from goalie.adjoint import AdjointSolver, annotate_qoi
from goalie.field import Field
from goalie.go_solver import GoalOrientedSolver
from goalie.log import WARNING
from goalie.mesh_seq import MeshSeq
from goalie.time_partition import TimeInterval, TimePartition


class BaseClasses:
    """
    Base classes for unit tests.
    """

    class RSpaceTestCase(unittest.TestCase):
        """
        Unit test case using R-space.
        """

        def setUp(self):
            mesh = UnitSquareMesh(1, 1)
            self.mesh_seq = MeshSeq([mesh])
            self.field = Field("field", family="Real", degree=0)
            self.function_space = FunctionSpace(mesh, self.field.get_element(mesh))

    class TrivialGoalOrientedBaseClass(unittest.TestCase):
        """
        Base class for tests with a trivial :class:`GoalOrientedMeshSeq`.
        """

        def setUp(self):
            self.meshes = [UnitSquareMesh(1, 1)]
            self.mesh_seq = MeshSeq(self.meshes)

        @staticmethod
        def constant_qoi(mesh_seq, solutions, index):
            R = FunctionSpace(mesh_seq[index], "R", 0)
            return lambda: Function(R).assign(1) * ufl.dx

        def go_solver(self, element=None, parameters=None):
            if element is None:
                element = FiniteElement("Real", ufl.triangle, 0)
            field = Field("field", finite_element=element)
            return GoalOrientedSolver(
                TimeInterval(1.0, [1.0], field),
                self.mesh_seq,
                qoi_type="steady",
                parameters=parameters,
            )

    class GoalOrientedBaseClass(unittest.TestCase):
        """
        Base class for tests with a complete :class:`GoalOrientedMeshSeq`.
        """

        def setUp(self):
            mesh = UnitSquareMesh(1, 1)
            self.mesh_seq = MeshSeq([mesh])
            self.field = Field("field", family="Real", degree=0)

        def go_solver(self, coeff_diff=0.0):
            self.time_partition = TimePartition(1.0, 1, 0.5, [self.field])
            outer_self = self

            class MySolver(GoalOrientedSolver):
                def get_initial_condition(self):
                    return {
                        outer_self.field.name: Function(
                            self.function_spaces[outer_self.field.name][0]
                        )
                    }

                def get_solver(self, index):
                    tp = outer_self.time_partition
                    R = FunctionSpace(outer_self.mesh_seq[index], "R", 0)
                    dt = Function(R).assign(tp.timesteps[index])

                    u, u_ = self.field_functions[outer_self.field.name]
                    f = Function(R).assign(1.0001)
                    v = TestFunction(u.function_space())
                    F = (u - u_) / dt * v * ufl.dx - f * v * ufl.dx
                    self.read_forms({outer_self.field.name: F})

                    for _ in range(tp.num_timesteps_per_subinterval[index]):
                        solve(F == 0, u, ad_block_tag=outer_self.field.name)
                        yield

                        u_.assign(u)
                        f += coeff_diff

                @annotate_qoi
                def get_qoi(mesh_seq, i):
                    def end_time_qoi():
                        u = mesh_seq.field_functions[outer_self.field.name][0]
                        return ufl.inner(u, u) * ufl.dx

                    return end_time_qoi

            return MySolver(
                self.time_partition,
                self.mesh_seq,
                qoi_type="end_time",
            )


class TestBlockLogic(BaseClasses.RSpaceTestCase):
    """
    Unit tests for :meth:`MeshSeq._dependency` and :meth:`MeshSeq._output`.
    """

    def setUp(self):
        super().setUp()
        self.solver = AdjointSolver(
            TimeInterval(1.0, 0.5, self.field),
            self.mesh_seq,
            qoi_type="end_time",
        )
        assert len(self.solver.meshes) == 1
        self.mesh = self.mesh_seq[0]

    def test_field_not_solved_for(self):
        field_not_solved_for = Field("field", family="Real", degree=0, solved_for=False)
        mesh_seq = AdjointSolver(
            TimeInterval(1.0, 0.5, field_not_solved_for),
            self.mesh_seq,
            qoi_type="end_time",
        )
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_solve_blocks("field", 0)
        msg = (
            "Cannot retrieve solve blocks for field 'field' because it isn't solved"
            " for."
        )
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_not_function(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(1)
        solve_block.get_outputs = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_wrong_function_space(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(
            Function(FunctionSpace(self.mesh, self.field.get_element(self.mesh)))
        )
        solve_block.get_outputs = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_wrong_name(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(self.function_space, name="field2"))
        solve_block.get_outputs = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._output("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no outputs."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_valid(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(self.function_space, name="field"))
        solve_block.get_outputs = lambda: [block_variable]
        self.assertIsNotNone(self.solver._output("field", 0, solve_block))

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_output_multiple_valid_error(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(self.function_space, name="field"))
        solve_block.get_outputs = lambda: [block_variable, block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._output("field", 0, solve_block)
        msg = (
            "Cannot determine a unique output index for the solution associated with"
            " field 'field' out of 2 candidates."
        )
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_not_function(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(1)
        solve_block.get_dependencies = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_wrong_function_space(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(
            Function(FunctionSpace(self.mesh, "Lagrange", 1))
        )
        solve_block.get_dependencies = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_wrong_name(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        function_space = FunctionSpace(self.mesh, "R", 0)
        block_variable = BlockVariable(Function(function_space, name="field_new"))
        solve_block.get_dependencies = lambda: [block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._dependency("field", 0, solve_block)
        msg = "Solve block for field 'field' on subinterval 0 has no dependencies."
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_valid(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(self.function_space, name="field_old"))
        solve_block.get_dependencies = lambda: [block_variable]
        self.assertIsNotNone(self.solver._dependency("field", 0, solve_block))

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_multiple_valid_error(self, MockSolveBlock):
        solve_block = MockSolveBlock()
        block_variable = BlockVariable(Function(self.function_space, name="field_old"))
        solve_block.get_dependencies = lambda: [block_variable, block_variable]
        with self.assertRaises(AttributeError) as cm:
            self.solver._dependency("field", 0, solve_block)
        msg = (
            "Cannot determine a unique dependency index for the lagged solution"
            " associated with field 'field' out of 2 candidates."
        )
        self.assertEqual(str(cm.exception), msg)

    @patch("firedrake.adjoint_utils.blocks.solving.GenericSolveBlock")
    def test_dependency_steady(self, MockSolveBlock):
        field = Field("field", family="Real", unsteady=False)
        self.time_interval = TimeInterval(1.0, 0.5, field)
        solver = AdjointSolver(
            self.time_interval,
            self.mesh_seq,
            qoi_type="end_time",
        )
        solve_block = MockSolveBlock()
        self.assertIsNone(solver._dependency("field", 0, solve_block))


class TestGetSolveBlocks(BaseClasses.RSpaceTestCase):
    """
    Unit tests for :meth:`get_solve_blocks`.
    """

    def setUp(self):
        super().setUp()
        time_interval = TimeInterval(1.0, [1.0], self.field)
        self.solver = AdjointSolver(
            time_interval,
            self.mesh_seq,
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
        solve(test * trial * ufl.dx == test * ufl.dx, sol, ad_block_tag=sol.name())

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def test_no_blocks(self):
        with self._caplog.at_level(logging.WARNING):
            blocks = self.solver.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = "Tape has no blocks!"
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_no_solve_blocks(self):
        fs = self.solver.function_spaces["field"][0]
        Function(fs).assign(1.0)
        with self._caplog.at_level(WARNING):
            blocks = self.solver.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = "Tape has no solve blocks!"
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_wrong_solve_block(self):
        fs = self.solver.function_spaces["field"][0]
        u = Function(fs, name="u")
        self.arbitrary_solve(u)
        with self._caplog.at_level(WARNING):
            blocks = self.solver.get_solve_blocks("field", 0)
        self.assertEqual(len(blocks), 0)
        self.assertEqual(len(self._caplog.records), 1)
        msg = (
            "No solve blocks associated with field 'field'."
            " Has ad_block_tag been used correctly?"
        )
        self.assertTrue(msg in str(self._caplog.records[0]))

    def test_wrong_function_space(self):
        fs = FunctionSpace(self.mesh_seq[0], "Lagrange", 1)
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        msg = (
            "Solve block list for field 'field' contains mismatching elements:"
            " <R0 on a triangle> vs. <CG1 on a triangle>."
        )
        with self.assertRaises(ValueError) as cm:
            self.solver.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)

    def test_too_many_timesteps(self):
        time_interval = TimeInterval(1.0, [0.5], self.field)
        solver = AdjointSolver(
            time_interval,
            MeshSeq([UnitSquareMesh(1, 1)]),
            qoi_type="end_time",
        )
        fs = solver.function_spaces["field"][0]
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        msg = (
            "Number of timesteps exceeds number of solve blocks for field 'field' on"
            " subinterval 0: 2 > 1."
        )
        with self.assertRaises(ValueError) as cm:
            solver.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)

    def test_incompatible_timesteps(self):
        time_interval = TimeInterval(1.0, [0.5], self.field)
        solver = AdjointSolver(
            time_interval,
            MeshSeq([UnitSquareMesh(1, 1)]),
            qoi_type="end_time",
        )
        fs = solver.function_spaces["field"][0]
        u = Function(fs, name="field")
        self.arbitrary_solve(u)
        self.arbitrary_solve(u)
        self.arbitrary_solve(u)
        msg = (
            "Number of timesteps is not divisible by number of solve blocks for field"
            " 'field' on subinterval 0: 2 vs. 3."
        )
        with self.assertRaises(ValueError) as cm:
            solver.get_solve_blocks("field", 0)
        self.assertEqual(str(cm.exception), msg)


class TestGoalOrientedMeshSeq(BaseClasses.TrivialGoalOrientedBaseClass):
    """
    Unit tests for a :class:`GoalOrientedMeshSeq`.
    """

    def test_read_forms_error_field(self):
        fields = [
            Field("field", family="R"),
            Field("field2", family="R", solved_for=False),
        ]
        go_solver = GoalOrientedSolver(
            TimeInterval(1.0, [1.0], fields),
            self.mesh_seq,
            qoi_type="steady",
        )

        with self.assertRaises(ValueError) as cm:
            go_solver.read_forms({"field2": None})
        msg = (
            "Unexpected field 'field2' in forms dictionary. Expected one of ['field']."
        )
        self.assertEqual(str(cm.exception), msg)

        with self.assertRaises(ValueError) as cm:
            go_solver.read_forms({"field3": None})
        msg = (
            "Unexpected field 'field3' in forms dictionary. Expected one of ['field']."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_read_forms_error_form(self):
        with self.assertRaises(TypeError) as cm:
            self.go_solver().read_forms({"field": None})
        msg = "Expected a UFL form for field 'field', not '<class 'NoneType'>'."
        self.assertEqual(str(cm.exception), msg)


class TestGlobalEnrichment(BaseClasses.TrivialGoalOrientedBaseClass):
    """
    Unit tests for global enrichment of a :class:`GoalOrientedMeshSeq`.
    """

    def element(self, family, degree, rank):
        if rank == 0:
            return FiniteElement(family, ufl.triangle, degree)
        elif rank == 1:
            return VectorElement(FiniteElement(family, ufl.triangle, degree))
        else:
            raise NotImplementedError

    def test_enrichment_error(self):
        with self.assertRaises(ValueError) as cm:
            self.go_solver().get_enriched_solver(enrichment_method="q")
        self.assertEqual(str(cm.exception), "Enrichment method 'q' not supported.")

    def test_num_enrichments_error(self):
        with self.assertRaises(ValueError) as cm:
            self.go_solver().get_enriched_solver(num_enrichments=0)
        msg = "A positive number of enrichments is required."
        self.assertEqual(str(cm.exception), msg)

    def test_form_error(self):
        with self.assertRaises(AttributeError) as cm:
            self.go_solver().forms()
        msg = (
            "Forms have not been read in. Use read_forms({'field_name': F}) in"
            " get_solver to read in the forms."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_h_enrichment_error(self):
        end_time = 1.0
        num_subintervals = 2
        dt = end_time / num_subintervals
        field = Field("field", family="Real")
        mesh_seq = GoalOrientedSolver(
            TimePartition(end_time, num_subintervals, dt, field),
            MeshSeq([UnitTriangleMesh()] * num_subintervals),
            get_qoi=self.constant_qoi,
            qoi_type="end_time",
        )
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_enriched_solver(enrichment_method="h")
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
        solver = self.go_solver()
        solver_e = solver.get_enriched_solver(
            enrichment_method="h", num_enrichments=num_enrichments
        )
        self.assertEqual(solver.meshes[0].num_cells(), 2)
        self.assertEqual(solver.meshes[0].num_vertices(), 4)
        self.assertEqual(solver.meshes[0].num_edges(), 5)
        n = num_enrichments
        self.assertEqual(solver_e.meshes[0].num_cells(), 2 * 4**n)
        self.assertEqual(solver_e.meshes[0].num_vertices(), (2 * n + 1) ** 2)
        self.assertEqual(
            solver_e.meshes[0].num_edges(),
            (2**n + 1) * (2 ** (n + 1)) + (2 ** (2 * n)),
        )

    @parameterized.expand(
        [
            ("Discontinuous Lagrange", 0, 0),
            ("Discontinuous Lagrange", 0, 1),
            ("Lagrange", 1, 0),
            ("Lagrange", 1, 1),
            ("Lagrange", 2, 0),
            ("Lagrange", 2, 1),
        ]
    )
    def test_h_enrichment_space(self, family, degree, rank):
        solver = self.go_solver(element=self.element(family, degree, rank))
        solver_e = solver.get_enriched_solver(enrichment_method="h", num_enrichments=1)
        field_name0 = solver.field_names[0]
        fspace = solver.function_spaces[field_name0][0]
        element = fspace.ufl_element()
        enriched_fspace = solver_e.function_spaces[field_name0][0]
        enriched_element = enriched_fspace.ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree(), enriched_element.degree())
        self.assertEqual(fspace.value_shape, enriched_fspace.value_shape)

    def test_p_enrichment_mesh(self):
        solver = self.go_solver(self.element("Lagrange", 1, 0))
        solver_e = solver.get_enriched_solver(enrichment_method="p", num_enrichments=1)
        self.assertEqual(self.meshes[0], solver.meshes[0])
        self.assertEqual(self.meshes[0], solver_e.meshes[0])

    @parameterized.expand(
        [
            ("Discontinuous Lagrange", 0, 0, 1),
            ("Discontinuous Lagrange", 0, 0, 2),
            ("Discontinuous Lagrange", 0, 1, 1),
            ("Discontinuous Lagrange", 0, 1, 2),
            ("Lagrange", 1, 0, 1),
            ("Lagrange", 1, 0, 2),
            ("Lagrange", 1, 1, 1),
            ("Lagrange", 1, 1, 2),
            ("Lagrange", 2, 0, 1),
            ("Lagrange", 2, 0, 2),
            ("Lagrange", 2, 1, 1),
            ("Lagrange", 2, 1, 2),
        ]
    )
    def test_p_enrichment_space(self, family, degree, rank, num_enrichments):
        solver = self.go_solver(element=self.element(family, degree, rank))
        solver_e = solver.get_enriched_solver(
            enrichment_method="p", num_enrichments=num_enrichments
        )
        field_name0 = solver.field_names[0]
        fspace = solver.function_spaces[field_name0][0]
        element = fspace.ufl_element()
        enriched_fspace = solver_e.function_spaces[field_name0][0]
        enriched_element = enriched_fspace.ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree() + num_enrichments, enriched_element.degree())
        self.assertEqual(fspace.value_shape, enriched_fspace.value_shape)

    @parameterized.expand(
        [
            ("Discontinuous Lagrange", 0, 0, "h", 1),
            ("Discontinuous Lagrange", 0, 0, "h", 2),
            ("Lagrange", 1, 0, "h", 1),
            ("Lagrange", 1, 0, "h", 2),
            ("Lagrange", 2, 0, "h", 1),
            ("Lagrange", 2, 0, "h", 2),
            ("Discontinuous Lagrange", 0, 0, "p", 1),
            ("Discontinuous Lagrange", 0, 0, "p", 2),
            ("Lagrange", 1, 0, "p", 1),
            ("Lagrange", 1, 0, "p", 2),
            ("Lagrange", 2, 0, "p", 1),
            ("Lagrange", 2, 0, "p", 2),
            ("Discontinuous Lagrange", 0, 1, "h", 1),
            ("Discontinuous Lagrange", 0, 1, "h", 2),
            ("Lagrange", 1, 1, "h", 1),
            ("Lagrange", 1, 1, "h", 2),
            ("Lagrange", 2, 1, "h", 1),
            ("Lagrange", 2, 1, "h", 2),
            ("Discontinuous Lagrange", 0, 1, "p", 1),
            ("Discontinuous Lagrange", 0, 1, "p", 2),
            ("Lagrange", 1, 1, "p", 1),
            ("Lagrange", 1, 1, "p", 2),
            ("Lagrange", 2, 1, "p", 1),
            ("Lagrange", 2, 1, "p", 2),
        ]
    )
    def test_enrichment_transfer(
        self, family, degree, rank, enrichment_method, num_enrichments
    ):
        solver = self.go_solver(element=self.element(family, degree, rank))
        solver_e = solver.get_enriched_solver(
            enrichment_method=enrichment_method, num_enrichments=num_enrichments
        )
        transfer = solver._get_transfer_function(enrichment_method)
        source = Function(solver.function_spaces["field"][0])
        x = ufl.SpatialCoordinate(solver.meshes[0])
        source.project(x if rank == 1 else sum(x))
        target = Function(solver_e.function_spaces["field"][0])
        transfer(source, target)
        self.assertAlmostEqual(norm(source), norm(target))


class TestDetectChangedCoefficients(BaseClasses.GoalOrientedBaseClass):
    """
    Unit tests for detecting changed coefficients using
    :meth:`GoalOrientedMeshSeq._detect_changing_coefficients`.
    """

    def test_constant_coefficients(self):
        solver = self.go_solver()
        # Solve over the first (only) subinterval
        next(solver._solve_adjoint(track_coefficients=True))
        # Check no coefficients have changed
        self.assertEqual(solver._changed_form_coeffs, {self.field.name: {}})

    def test_changed_coefficients(self):
        # Change coefficient f by coeff_diff every timestep
        coeff_diff = 1.1
        solver = self.go_solver(coeff_diff=coeff_diff)
        # Solve over the first (only) subinterval
        next(solver._solve_adjoint(track_coefficients=True))
        changed_coeffs_dict = solver._changed_form_coeffs[self.field.name]
        coeff_idx = next(iter(changed_coeffs_dict))
        for export_idx, f in changed_coeffs_dict[coeff_idx].items():
            self.assertTrue(f.vector().gather() == [1.0001 + export_idx * coeff_diff])
