"""
Unit tests for :class:`~.FunctionData` and its subclasses.
"""

import abc
import os
import unittest
from tempfile import TemporaryDirectory

from firedrake.function import Function
from firedrake.functionspace import FunctionSpace
from firedrake.mg.mesh import MeshHierarchy
from firedrake.utility_meshes import UnitTriangleMesh

from goalie.function_data import AdjointSolutionData, ForwardSolutionData, IndicatorData
from goalie.time_partition import TimePartition
from goalie.utility import AttrDict


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
        for sub_data in data:
            self.assertTrue(isinstance(sub_data, AttrDict))
            self.assertTrue(self.field in sub_data)
            self.assertTrue(isinstance(sub_data[self.field], list))
            for f in sub_data[self.field]:
                self.assertTrue(isinstance(f, Function))


class TestExportFunctionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for exporting and checkpointing :class:`~.FunctionData`.
    """

    def setUp(self):
        super().setUpUnsteady()
        self.labels = ("forward", "forward_old")

    def _create_function_data(self):
        self.solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        self.solution_data._create_data()

    def test_export_extension_error(self):
        with self.assertRaises(ValueError) as cm:
            self.solution_data.export("test.ext")
        msg = (
            "Output file format not recognised: 'test.ext'."
            + " Supported formats are '.pvd' and '.h5'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_export_field_error(self):
        with self.assertRaises(ValueError) as cm:
            self.solution_data.export("test.pvd", export_field_types="test")
        msg = (
            "Field types ['test'] not recognised."
            + f" Available types are {self.solution_data.labels}."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_export_pvd(self):
        with TemporaryDirectory() as tmpdir:
            export_filepath = os.path.join(tmpdir, "test.pvd")
            self.solution_data.export(export_filepath)
            self.assertTrue(os.path.exists(export_filepath))

    def test_export_pvd_ic(self):
        ic = {field: Function(fs[0]) for field, fs in self.function_spaces.items()}
        with TemporaryDirectory() as tmpdir:
            export_filepath = os.path.join(tmpdir, "test.pvd")
            self.solution_data.export(export_filepath, initial_condition=ic)
            self.assertTrue(os.path.exists(export_filepath))

    def test_export_h5(self):
        with TemporaryDirectory() as tmpdir:
            export_filepath = os.path.join(tmpdir, "test.h5")
            self.solution_data.export(export_filepath)
            self.assertTrue(os.path.exists(export_filepath))


class TestTransferFunctionData(BaseTestCases.TestFunctionData):
    """
    Unit tests for transferring data from one :class:`~.FunctionData` to another.
    """

    def setUp(self):
        super().setUpUnsteady()
        self.labels = ("forward", "forward_old")

    def _create_function_data(self):
        self.solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        self.solution_data._create_data()

        # Assign 1 to all functions
        tp = self.solution_data.time_partition
        for field in tp.field_names:
            for label in self.solution_data.labels:
                for i in range(tp.num_subintervals):
                    for j in range(tp.num_exports_per_subinterval[i] - 1):
                        self.solution_data._data[field][label][i][j].assign(1)

    def test_transfer_method_error(self):
        target_solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        target_solution_data._create_data()
        with self.assertRaises(ValueError) as cm:
            self.solution_data.transfer(target_solution_data, method="invalid_method")
        self.assertEqual(
            str(cm.exception),
            "Transfer method 'invalid_method' not supported."
            " Supported methods are 'interpolate', 'project', and 'prolong'.",
        )

    def test_transfer_subintervals_error(self):
        target_time_partition = TimePartition(
            1.5 * self.time_partition.end_time,
            self.time_partition.num_subintervals + 1,
            self.time_partition.timesteps + [0.25],
            self.time_partition.field_names,
        )
        target_function_spaces = {
            self.field: [
                FunctionSpace(self.mesh, "DG", 0)
                for _ in range(target_time_partition.num_subintervals)
            ]
        }
        target_solution_data = ForwardSolutionData(
            target_time_partition, target_function_spaces
        )
        target_solution_data._create_data()
        with self.assertRaises(ValueError) as cm:
            self.solution_data.transfer(target_solution_data, method="interpolate")
        self.assertEqual(
            str(cm.exception),
            "Source and target have different numbers of subintervals.",
        )

    def test_transfer_exports_error(self):
        target_time_partition = TimePartition(
            self.time_partition.end_time,
            self.time_partition.num_subintervals,
            self.time_partition.timesteps,
            self.time_partition.field_names,
            num_timesteps_per_export=[1, 2],
        )
        target_function_spaces = {
            self.field: [
                FunctionSpace(self.mesh, "DG", 0)
                for _ in range(target_time_partition.num_subintervals)
            ]
        }
        target_solution_data = ForwardSolutionData(
            target_time_partition, target_function_spaces
        )
        target_solution_data._create_data()
        with self.assertRaises(ValueError) as cm:
            self.solution_data.transfer(target_solution_data, method="interpolate")
        self.assertEqual(
            str(cm.exception),
            "Source and target have different numbers of exports per subinterval.",
        )

    def test_transfer_common_fields_error(self):
        target_time_partition = TimePartition(
            self.time_partition.end_time,
            self.time_partition.num_subintervals,
            self.time_partition.timesteps,
            ["different_field"],
        )
        target_function_spaces = {
            "different_field": [
                FunctionSpace(self.mesh, "DG", 0)
                for _ in range(target_time_partition.num_subintervals)
            ]
        }
        target_solution_data = ForwardSolutionData(
            target_time_partition, target_function_spaces
        )
        target_solution_data._create_data()
        with self.assertRaises(ValueError) as cm:
            self.solution_data.transfer(target_solution_data, method="interpolate")
        self.assertEqual(
            str(cm.exception), "No common fields between source and target."
        )

    def test_transfer_common_labels_error(self):
        target_solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        target_solution_data._create_data()
        target_solution_data.labels = ("different_label",)
        with self.assertRaises(ValueError) as cm:
            self.solution_data.transfer(target_solution_data, method="interpolate")
        self.assertEqual(
            str(cm.exception), "No common labels between source and target."
        )

    def test_transfer_interpolate(self):
        target_solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        target_solution_data._create_data()
        self.solution_data.transfer(target_solution_data, method="interpolate")
        for field in self.solution_data.time_partition.field_names:
            for label in self.solution_data.labels:
                for i in range(self.solution_data.time_partition.num_subintervals):
                    for j in range(
                        self.solution_data.time_partition.num_exports_per_subinterval[i]
                        - 1
                    ):
                        source_function = self.solution_data._data[field][label][i][j]
                        target_function = target_solution_data._data[field][label][i][j]
                        self.assertTrue(
                            source_function.dat.data.all()
                            == target_function.dat.data.all()
                        )

    def test_transfer_project(self):
        target_solution_data = ForwardSolutionData(
            self.time_partition, self.function_spaces
        )
        target_solution_data._create_data()
        self.solution_data.transfer(target_solution_data, method="project")
        for field in self.solution_data.time_partition.field_names:
            for label in self.solution_data.labels:
                for i in range(self.solution_data.time_partition.num_subintervals):
                    for j in range(
                        self.solution_data.time_partition.num_exports_per_subinterval[i]
                        - 1
                    ):
                        source_function = self.solution_data._data[field][label][i][j]
                        target_function = target_solution_data._data[field][label][i][j]
                        self.assertTrue(
                            source_function.dat.data.all()
                            == target_function.dat.data.all()
                        )

    def test_transfer_prolong(self):
        enriched_mesh = MeshHierarchy(self.mesh, 1)[-1]
        target_function_spaces = {
            self.field: [
                FunctionSpace(enriched_mesh, "DG", 0)
                for _ in range(self.num_subintervals)
            ]
        }
        target_solution_data = ForwardSolutionData(
            self.time_partition, target_function_spaces
        )
        target_solution_data._create_data()
        self.solution_data.transfer(target_solution_data, method="prolong")
        for field in self.solution_data.time_partition.field_names:
            for label in self.solution_data.labels:
                for i in range(self.solution_data.time_partition.num_subintervals):
                    for j in range(
                        self.solution_data.time_partition.num_exports_per_subinterval[i]
                        - 1
                    ):
                        source_function = self.solution_data._data[field][label][i][j]
                        target_function = target_solution_data._data[field][label][i][j]
                        self.assertTrue(
                            source_function.dat.data.all()
                            == target_function.dat.data.all()
                        )
