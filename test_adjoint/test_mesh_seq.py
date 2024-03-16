"""
Testing for the mesh sequence objects.
"""
from firedrake import *
from goalie_adjoint import *
from goalie.log import *
from goalie.mesh_seq import MeshSeq
from goalie.go_mesh_seq import GoalOrientedMeshSeq
from goalie.time_partition import TimeInterval
from parameterized import parameterized
import logging
import pyadjoint
import pytest
import unittest


class TestGetSolveBlocks(unittest.TestCase):
    """
    Unit tests for :meth:`get_solve_blocks`.
    """

    @staticmethod
    def get_function_spaces(mesh):
        return {"field": FunctionSpace(mesh, "R", 0)}

    def setUp(self):
        time_interval = TimeInterval(1.0, [1.0], ["field"])
        self.mesh_seq = MeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
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
        mesh_seq = MeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
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
        mesh_seq = MeshSeq(
            time_interval,
            [UnitSquareMesh(1, 1)],
            get_function_spaces=self.get_function_spaces,
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
        msg = "Enrichment method 'q' not supported."
        self.assertEqual(str(cm.exception), msg)

    def test_num_enrichments_error(self):
        mesh_seq = self.go_mesh_seq(self.get_function_spaces_decorator("R", 0, 0))
        with self.assertRaises(ValueError) as cm:
            mesh_seq.get_enriched_mesh_seq(num_enrichments=0)
        msg = "A positive number of enrichments is required."
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
        element = mesh_seq.function_spaces[self.field][0].ufl_element()
        enriched_element = mesh_seq_e.function_spaces[self.field][0].ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree(), enriched_element.degree())
        self.assertEqual(element.value_shape, enriched_element.value_shape)

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
        element = mesh_seq.function_spaces[self.field][0].ufl_element()
        enriched_element = mesh_seq_e.function_spaces[self.field][0].ufl_element()
        self.assertEqual(element.family(), enriched_element.family())
        self.assertEqual(element.degree() + num_enrichments, enriched_element.degree())
        self.assertEqual(element.value_shape, enriched_element.value_shape)

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


class TestErrorIndication(TrivialGoalOrientedBaseClass):
    """
    Unit tests for :meth:`indicate_errors`.
    """

    def test_form_error(self):
        mesh_seq = GoalOrientedMeshSeq(
            TimeInstant([]),
            UnitTriangleMesh(),
            get_qoi=self.constant_qoi,
            qoi_type="steady",
        )
        mesh_seq._get_function_spaces = lambda _: {}
        mesh_seq._get_form = lambda _: lambda *_: 0
        mesh_seq._get_solver = lambda _: lambda *_: {}
        with self.assertRaises(TypeError) as cm:
            mesh_seq.fixed_point_iteration(lambda *_: [False])
        msg = "The function defined by get_form should return a dictionary, not type '<class 'int'>'."
        self.assertEqual(str(cm.exception), msg)
