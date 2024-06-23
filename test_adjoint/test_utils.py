import unittest

import numpy as np
from firedrake import *

from goalie.adjoint import annotate_qoi
from goalie_adjoint import *


class TestAdjointUtils(unittest.TestCase):
    """
    Unit tests for adjoint utils.
    """

    def setUp(self):
        self.time_interval = TimeInterval(1.0, [0.5], ["field"])
        self.mesh = UnitSquareMesh(1, 1)

    def mesh_seq(self, qoi_type="end_time"):
        return AdjointMeshSeq(self.time_interval, [self.mesh], qoi_type=qoi_type)

    def test_annotate_qoi_0args(self):
        @annotate_qoi
        def get_qoi(mesh_seq, i):
            R = FunctionSpace(mesh_seq[i], "R", 0)

            def qoi():
                return Function(R).assign(1) * dx

            return qoi

        get_qoi(self.mesh_seq("end_time"), 0)

    def test_annotate_qoi_1arg(self):
        @annotate_qoi
        def get_qoi(mesh_seq, i):
            R = FunctionSpace(mesh_seq[i], "R", 0)

            def qoi(t):
                return Function(R).assign(1) * dx

            return qoi

        get_qoi(self.mesh_seq("time_integrated"), 0)

    def test_annotate_qoi_0args_error(self):
        @annotate_qoi
        def get_qoi(mesh_seq, i):
            R = FunctionSpace(mesh_seq[i], "R", 0)

            def qoi():
                return Function(R).assign(1) * dx

            return qoi

        with self.assertRaises(ValueError) as cm:
            get_qoi(self.mesh_seq("time_integrated"), 0)
        msg = "Expected qoi_type to be 'end_time' or 'steady', not 'time_integrated'."
        assert str(cm.exception) == msg

    def test_annotate_qoi_1arg_error(self):
        @annotate_qoi
        def get_qoi(mesh_seq, i):
            R = FunctionSpace(mesh_seq[i], "R", 0)

            def qoi(t):
                return Function(R).assign(1) * dx

            return qoi

        with self.assertRaises(ValueError) as cm:
            get_qoi(self.mesh_seq("end_time"), 0)
        msg = "Expected qoi_type to be 'time_integrated', not 'end_time'."
        assert str(cm.exception) == msg

    def test_annotate_qoi_2args_error(self):
        @annotate_qoi
        def get_qoi(mesh_seq, i):
            R = FunctionSpace(mesh_seq[i], "R", 0)

            def qoi(t, r):
                return Function(R).assign(1) * dx

            return qoi

        with self.assertRaises(ValueError) as cm:
            get_qoi(self.mesh_seq("time_integrated"), 0)
        assert str(cm.exception) == "QoI should have 0 or 1 args, not 2."

    def test_annotate_qoi_not_steady(self):
        with self.assertRaises(ValueError) as cm:
            self.mesh_seq("steady")
        msg = "QoI type is set to 'steady' but the time partition is not steady."
        assert str(cm.exception) == msg

    def test_annotate_qoi_steady(self):
        time_interval = TimeInterval(1.0, [1.0], ["field"])
        with self.assertRaises(ValueError) as cm:
            AdjointMeshSeq(time_interval, [self.mesh], qoi_type="end_time")
        msg = "Time partition is steady but the QoI type is set to 'end_time'."
        assert str(cm.exception) == msg

    def test_qoi_type_error(self):
        with self.assertRaises(ValueError) as cm:
            self.mesh_seq("qoi_type")
        msg = (
            "QoI type 'qoi_type' not recognised."
            " Choose from 'end_time', 'time_integrated', or 'steady'."
        )
        assert str(cm.exception) == msg

    def test_qoi_notimplemented(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.mesh_seq("end_time").get_qoi(0)
        assert str(cm.exception) == "'get_qoi' is not implemented."

    def test_qoi_convergence_values_lt_2(self):
        mesh_seq = self.mesh_seq("end_time")
        mesh_seq.fp_iteration = mesh_seq.params.miniter + 1
        mesh_seq.qoi_values = [1.0]
        self.assertFalse(mesh_seq.check_qoi_convergence())

    def test_qoi_convergence_values_lt_miniter(self):
        mesh_seq = self.mesh_seq("end_time")
        mesh_seq.fp_iteration = mesh_seq.params.miniter
        mesh_seq.qoi_values = np.ones(mesh_seq.params.miniter - 1)
        self.assertFalse(mesh_seq.check_qoi_convergence())

    def test_qoi_convergence_values_constant(self):
        mesh_seq = self.mesh_seq("end_time")
        mesh_seq.fp_iteration = mesh_seq.params.miniter
        mesh_seq.qoi_values = np.ones(mesh_seq.params.miniter + 1)
        self.assertTrue(mesh_seq.check_qoi_convergence())

    def test_qoi_convergence_values_not_converged(self):
        mesh_seq = self.mesh_seq("end_time")
        mesh_seq.fp_iteration = mesh_seq.params.miniter
        mesh_seq.qoi_values = np.ones(mesh_seq.params.miniter + 1)
        mesh_seq.qoi_values[-1] = 100.0
        self.assertFalse(mesh_seq.check_qoi_convergence())
