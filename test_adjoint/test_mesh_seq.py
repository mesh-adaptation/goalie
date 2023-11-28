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
import unittest
from setup_adjoint_tests import *


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


class TestGlobalEnrichment(unittest.TestCase):
    """
    Unit tests for global enrichment of a :class:`GoalOrientedMeshSeq`.
    """

    def setUp(self):
        self.field = "field"
        self.time_interval = TimeInterval(1.0, [1.0], [self.field])
        self.meshes = [UnitSquareMesh(1, 1)]

    def go_mesh_seq(self, get_function_spaces):
        return GoalOrientedMeshSeq(
            self.time_interval,
            self.meshes,
            get_function_spaces=get_function_spaces,
            get_form=empty_get_form,
            get_bcs=empty_get_bcs,
            get_solver=empty_get_solver,
            qoi_type="steady",
        )

    def get_function_spaces_decorator(self, degree, family, rank):
        def get_function_spaces(mesh):
            if rank == 0:
                return {self.field: FunctionSpace(mesh, degree, family)}
            elif rank == 1:
                return {self.field: VectorFunctionSpace(mesh, degree, family)}
            else:
                raise NotImplementedError

        return get_function_spaces

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
        element = mesh_seq.function_spaces[self.field][0].ufl_element()
        enriched_element = mesh_seq_e.function_spaces[self.field][0].ufl_element()
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
