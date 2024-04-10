"""
Unit tests for :class:`~.FunctionData` and its subclasses.
"""

from firedrake import *
from goalie import *
import abc
import unittest


class BaseTestCases:
    """
    Class containing abstract base classes for unit testing subclasses of
    :class:`~.FunctionData`.
    """

    class TestFunctionData(unittest.TestCase, abc.ABC):
        """
        Base class for unit testing subclasses of :class:`~.FunctionData`.
        """

        def setUpUnsteady(self):
            end_time = 1.0
            self.num_subintervals = 2
            timesteps = [0.5, 0.25]
            self.field = "field"
            self.num_exports = [1, 2]
            self.mesh = UnitTriangleMesh()
            self.time_partition = TimePartition(
                end_time, self.num_subintervals, timesteps, self.field
            )
            self.function_spaces = {
                self.field: [
                    FunctionSpace(self.mesh, "DG", 0)
                    for _ in range(self.num_subintervals)
                ]
            }
            self._create_function_data()

        def setUpSteady(self):
            end_time = 1.0
            self.num_subintervals = 1
            timesteps = [1.0]
            self.field = "field"
            self.num_exports = [1]
            self.mesh = UnitTriangleMesh()
            self.time_partition = TimePartition(
                end_time, self.num_subintervals, timesteps, self.field
            )
            self.function_spaces = {
                self.field: [
                    FunctionSpace(self.mesh, "DG", 0)
                    for _ in range(self.num_subintervals)
                ]
            }
            self._create_function_data()

        @abc.abstractmethod
        def _create_function_data(self):
            pass

        def test_extract_by_field(self):
            data = self.solution_data.extract(layout="field")
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

        def test_extract_by_label(self):
            data = self.solution_data.extract(layout="label")
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

        def test_extract_by_subinterval(self):
            data = self.solution_data.extract(layout="subinterval")
            self.assertTrue(isinstance(data, list))
            self.assertEqual(len(data), self.num_subintervals)
            for i, sub_data in enumerate(data):
                self.assertTrue(isinstance(sub_data, AttrDict))
                self.assertTrue(self.field in sub_data)
                self.assertTrue(isinstance(sub_data[self.field], AttrDict))
                for label in self.labels:
                    self.assertTrue(label in sub_data[self.field])
                    self.assertTrue(isinstance(sub_data[self.field][label], list))
                    self.assertEqual(
                        len(sub_data[self.field][label]), self.num_exports[i]
                    )
                    for f in sub_data[self.field][label]:
                        self.assertTrue(isinstance(f, Function))


class TestSteadyForwardSolutionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for :class:`~.ForwardSolutionData`.
    """

    def setUp(self):
        super().setUpSteady()
        self.labels = ("forward",)

    def _create_function_data(self):
        self.solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )


class TestUnsteadyForwardSolutionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for :class:`~.ForwardSolutionData`.
    """

    def setUp(self):
        super().setUpUnsteady()
        self.labels = ("forward", "forward_old")

    def _create_function_data(self):
        self.solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )


class TestSteadyAdjointSolutionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for :class:`~.AdjointSolutionData`.
    """

    def setUp(self):
        super().setUpSteady()
        self.labels = ("forward", "adjoint")

    def _create_function_data(self):
        self.solution_data = AdjointSolutionData(
            self.time_partition, self.function_spaces
        )


class TestUnsteadyAdjointSolutionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for :class:`~.AdjointSolutionData`.
    """

    def setUp(self):
        super().setUpUnsteady()
        self.labels = ("forward", "forward_old", "adjoint", "adjoint_next")

    def _create_function_data(self):
        self.solution_data = AdjointSolutionData(
            self.time_partition, self.function_spaces
        )


class TestIndicatorData(BaseTestCases.TestFunctionData):
    """
    Unit tests for :class:`~.Indicatordata`.
    """

    def setUp(self):
        super().setUpUnsteady()
        self.labels = ("error_indicator",)

    def _create_function_data(self):
        self.solution_data = IndicatorData(
            self.time_partition, [self.mesh for _ in range(self.num_subintervals)]
        )

    def _test_extract_by_field_or_label(self, data):
        self.assertTrue(isinstance(data, AttrDict))
        self.assertTrue(self.field in data)
        self.assertEqual(len(data[self.field]), self.num_subintervals)
        for i, num_exports in enumerate(self.num_exports):
            self.assertTrue(isinstance(data[self.field][i], list))
            self.assertEqual(len(data[self.field][i]), num_exports)
            for f in data[self.field][i]:
                self.assertTrue(isinstance(f, Function))

    def test_extract_by_field(self):
        data = self.solution_data.extract(layout="field")
        self._test_extract_by_field_or_label(data)

    def test_extract_by_label(self):
        data = self.solution_data.extract(layout="label")
        self._test_extract_by_field_or_label(data)

    def test_extract_by_subinterval(self):
        data = self.solution_data.extract(layout="subinterval")
        self.assertTrue(isinstance(data, list))
        self.assertEqual(len(data), self.num_subintervals)
        for i, sub_data in enumerate(data):
            self.assertTrue(isinstance(sub_data, AttrDict))
            self.assertTrue(self.field in sub_data)
            self.assertTrue(isinstance(sub_data[self.field], list))
            for f in sub_data[self.field]:
                self.assertTrue(isinstance(f, Function))


if __name__ == "__main__":
    unittest.main()
