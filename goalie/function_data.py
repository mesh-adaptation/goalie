r"""
Nested dictionaries of solution data :class:`~.Function`\s.
"""
import firedrake.function as ffunc
import firedrake.functionspace as ffs
from .utility import AttrDict
import abc

__all__ = [
    "SteadyForwardSolutionData",
    "UnsteadyForwardSolutionData",
    "ForwardSolutionData",
    "SteadyAdjointSolutionData",
    "UnsteadyAdjointSolutionData",
    "AdjointSolutionData",
    "IndicatorData",
]


class FunctionData(abc.ABC):
    """
    Abstract base class for classes holding field data.
    """

    labels = None

    def __init__(self, time_partition, function_spaces):
        r"""
        :arg time_partition: the :class:`~.TimePartition` used to discretise the problem
            in time
        :arg function_spaces: the dictionary of :class:`~.FunctionSpace`\s used to
            discretise the problem in space
        """
        self.time_partition = time_partition
        self.function_spaces = function_spaces
        self._data = None
        self._create_data()

    def _create_data(self):
        assert self.labels is not None
        P = self.time_partition
        self._data = AttrDict(
            {
                field: AttrDict(
                    {
                        label: [
                            [
                                ffunc.Function(fs, name=f"{field}_{label}")
                                for j in range(P.num_exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(self.function_spaces[field])
                        ]
                        for label in self.labels
                    }
                )
                for field in P.fields
            }
        )

    @property
    def data(self):
        if self._data is None:
            self._create_data()
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def items(self):
        return self.data.items()


class SolutionData(FunctionData, abc.ABC):
    """
    Abstract base class that defines the API for solution data classes.
    """

    @property
    def solutions(self):
        return self.data


class SteadyForwardSolutionData(SolutionData):
    """
    Class representing solution data for steady-state forward problems.
    """

    labels = ("forward",)


class UnsteadyForwardSolutionData(SolutionData):
    """
    Class representing solution data for time-dependent forward problems.
    """

    labels = ("forward", "forward_old")


class ForwardSolutionData(UnsteadyForwardSolutionData):
    """
    Class representing solution data for general forward problems.
    """


class SteadyAdjointSolutionData(SolutionData):
    """
    Class representing solution data for steady-state adjoint problems.
    """

    labels = ("forward", "forward_old", "adjoint")


class UnsteadyAdjointSolutionData(SolutionData):
    """
    Class representing solution data for time-dependent adjoint problems.
    """

    labels = ("forward", "forward_old", "adjoint", "adjoint_next")


class AdjointSolutionData(UnsteadyAdjointSolutionData):
    """
    Class representing solution data for general adjoint problems.
    """


class IndicatorData(FunctionData):
    """
    Class representing error indicator data.

    Note that this class has a single dictionary with the field name as the key, rather
    than a doubly-nested dictionary.
    """

    labels = ("error_indicator",)

    def __init__(self, time_partition, meshes):
        """
        :arg time_partition: the :class:`~.TimePartition` used to discretise the problem
            in time
        :arg meshes: the list of meshes used to discretise the problem in space
        """
        P0_spaces = [ffs.FunctionSpace(mesh, "DG", 0) for mesh in meshes]
        super().__init__(
            time_partition, {key: P0_spaces for key in time_partition.fields}
        )

    def _create_data(self):
        assert len(self.labels) == 1
        super()._create_data()
        self._data = AttrDict(
            {
                field: self.data[field][self.labels[0]]
                for field in self.time_partition.fields
            }
        )

    @property
    def indicators(self):
        return self.data
