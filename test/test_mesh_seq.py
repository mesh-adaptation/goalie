"""
Testing for the mesh sequence objects.
"""

import re
import unittest

from firedrake import (
    UnitCubeMesh,
    UnitIntervalMesh,
    UnitSquareMesh,
)
from parameterized import parameterized

from goalie.mesh_seq import MeshSeq


class BaseClasses:
    """
    Base classes for mesh sequence unit testing.
    """

    class MeshSeqTestCase(unittest.TestCase):
        """
        Test case with a simple mesh constructor.
        """

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
            MeshSeq(meshes)
        msg = "Meshes must all have the same topological dimension."
        self.assertEqual(str(cm.exception), msg)

    @parameterized.expand([1, 3])
    def test_plot_dim_error(self, dim):
        mesh_seq = MeshSeq(self.trivial_mesh(dim))
        with self.assertRaises(ValueError) as cm:
            mesh_seq.plot()
        self.assertEqual(str(cm.exception), "MeshSeq plotting only supported in 2D.")


class TestGeneric(BaseClasses.MeshSeqTestCase):
    """
    Generic unit tests for :class:`MeshSeq`.
    """

    def test_setitem(self):
        mesh1 = UnitSquareMesh(1, 1, diagonal="left")
        mesh2 = UnitSquareMesh(1, 1, diagonal="right")
        mesh_seq = MeshSeq([mesh1])
        self.assertEqual(mesh_seq[0], mesh1)
        mesh_seq[0] = mesh2
        self.assertEqual(mesh_seq[0], mesh2)

    def test_counting_2d(self):
        mesh_seq = MeshSeq([UnitSquareMesh(3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [18])
        self.assertEqual(mesh_seq.count_vertices(), [16])

    def test_counting_3d(self):
        mesh_seq = MeshSeq([UnitCubeMesh(3, 3, 3)])
        self.assertEqual(mesh_seq.count_elements(), [162])
        self.assertEqual(mesh_seq.count_vertices(), [64])


class TestStringFormatting(BaseClasses.MeshSeqTestCase):
    """
    Test that the :meth:`__str__` and :meth:`__repr__` methods work as intended for
    Goalie's :class:`MeshSeq` object.
    """

    def test_mesh_seq_str(self):
        mesh_seq = MeshSeq([UnitSquareMesh(1, 1)])
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>']")

    def test_mesh_seq_2_str(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(meshes)
        got = re.sub("#[0-9]*", "?", str(mesh_seq))
        self.assertEqual(got, "['<Mesh ?>', '<Mesh ?>']")

    def test_mesh_seq_repr(self):
        mesh_seq = MeshSeq([UnitSquareMesh(1, 1)])
        expected = (
            "MeshSeq([Mesh(VectorElement("
            "FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_2_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
        ]
        mesh_seq = MeshSeq(meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))

    def test_mesh_seq_3_repr(self):
        meshes = [
            UnitSquareMesh(1, 1, diagonal="left"),
            UnitSquareMesh(1, 1, diagonal="right"),
            UnitSquareMesh(1, 1, diagonal="left"),
        ]
        mesh_seq = MeshSeq(meshes)
        expected = (
            "MeshSeq(["
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*), "
            "..."
            "Mesh(VectorElement(FiniteElement('Lagrange', triangle, 1), dim=2), .*)])"
        )
        self.assertTrue(re.match(repr(mesh_seq), expected))
