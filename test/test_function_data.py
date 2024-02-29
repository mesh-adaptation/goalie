"""
Unit tests for :class:`~.FunctionData` and its subclasses.
"""
from firedrake import *
from goalie import *
import unittest


class TestForwardSolutionData(unittest.TestCase):
    """
    Unit tests for :class:`~.ForwardSolutionData`.
    """

    def setUp(self):
        end_time = 1.0
        self.num_subintervals = 2
        timesteps = [0.5, 0.25]
        self.field = "field"
        self.labels = ("forward", "forward_old")
        self.num_exports = [1, 2]
        mesh = UnitTriangleMesh()
        self.time_partition = TimePartition(
            end_time, self.num_subintervals, timesteps, self.field
        )
        self.function_spaces = {
            self.field: [
                FunctionSpace(mesh, "DG", 0) for _ in range(self.num_subintervals)
            ]
        }
        self._create_solution_data()

    def _create_solution_data(self):
        self.solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )

    def test_data_by_field(self):
        data = self.solution_data.data_by_field
        self.assertTrue(isinstance(data, AttrDict))
        self.assertTrue(self.field in data)
        for label in self.labels:
            self.assertTrue(isinstance(data[self.field], AttrDict))
            self.assertTrue(label in data[self.field])
            self.assertTrue(isinstance(data[self.field][label], list))
            self.assertEqual(len(data[self.field][label]), self.num_subintervals)
            for i, num_exports in enumerate(self.num_exports):
                self.assertTrue(isinstance(data[self.field][label][i], list))
                self.assertEqual(len(data[self.field][label][i]), num_exports)
                for f in data[self.field][label][i]:
                    self.assertTrue(isinstance(f, Function))

    def test_data_by_label(self):
        data = self.solution_data.data_by_label
        self.assertTrue(isinstance(data, AttrDict))
        for label in self.labels:
            self.assertTrue(label in data)
            self.assertTrue(isinstance(data[label], AttrDict))
            self.assertTrue(self.field in data[label])
            self.assertTrue(isinstance(data[label][self.field], list))
            self.assertEqual(len(data[label][self.field]), self.num_subintervals)
            for i, num_exports in enumerate(self.num_exports):
                self.assertTrue(isinstance(data[label][self.field][i], list))
                self.assertEqual(len(data[label][self.field][i]), num_exports)
                for f in data[label][self.field][i]:
                    self.assertTrue(isinstance(f, Function))

    def test_data_by_subinterval(self):
        data = self.solution_data.data_by_subinterval
        self.assertTrue(isinstance(data, list))
        self.assertEqual(len(data), self.num_subintervals)
        for i, sub_data in enumerate(data):
            self.assertTrue(isinstance(sub_data, AttrDict))
            self.assertTrue(self.field in sub_data)
            self.assertTrue(isinstance(sub_data[self.field], AttrDict))
            for label in self.labels:
                self.assertTrue(label in sub_data[self.field])
                self.assertTrue(isinstance(sub_data[self.field][label], list))
                self.assertEqual(len(sub_data[self.field][label]), self.num_exports[i])
                for f in sub_data[self.field][label]:
                    self.assertTrue(isinstance(f, Function))
