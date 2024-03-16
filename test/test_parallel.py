from goalie.mesh_seq import MeshSeq
from goalie.time_partition import TimeInterval
from firedrake import *
import pytest


@pytest.mark.parallel(nprocs=2)
def test_counting_parallel():
    assert COMM_WORLD.size == 2
    time_interval = TimeInterval(1.0, [0.5], ["field"])
    mesh_seq = MeshSeq(time_interval, [UnitSquareMesh(3, 3)])
    assert mesh_seq.count_elements() == [18]
    assert mesh_seq.count_vertices() == [16]
