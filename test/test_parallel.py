import pytest
from firedrake.utility_meshes import UnitCubeMesh, UnitSquareMesh
from pyop2.mpi import COMM_WORLD

from goalie.mesh_seq import MeshSeq


@pytest.mark.parallel(nprocs=2)
def test_counting_2d():
    assert COMM_WORLD.size == 2
    mesh_seq = MeshSeq([UnitSquareMesh(3, 3)])
    assert mesh_seq.count_elements() == [18]
    assert mesh_seq.count_vertices() == [16]


@pytest.mark.parallel(nprocs=2)
def test_counting_3d():
    assert COMM_WORLD.size == 2
    mesh_seq = MeshSeq([UnitCubeMesh(3, 3, 3)])
    assert mesh_seq.count_elements() == [162]
    assert mesh_seq.count_vertices() == [64]
