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


class SolutionData(abc.ABC):
    """
    Abstract base class that defines the API for solution data classes.
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
        self._solutions = None
        self._create_solutions()

    def _create_solutions(self):
        assert self.labels is not None
        P = self.time_partition
        self._solutions = AttrDict(
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
    def solutions(self):
        if self._solutions is None:
            self._create_solutions()
        return self._solutions

    def __getitem__(self, key):
        return self.solutions[key]


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


class IndicatorData:
    """
    Class representing error indicator data.
    """

    def __init__(self, time_partition, meshes):
        """
        :arg time_partition: the :class:`~.TimePartition` used to discretise the problem
            in time
        :arg meshes: the list of meshes used to discretise the problem in space
        """
        self.time_partition = time_partition
        self.meshes = meshes
        self._indicators = None
        self._create_indicators()

    def _create_indicators(self):
        P0_spaces = [ffs.FunctionSpace(mesh, "DG", 0) for mesh in self.meshes]
        P = self.time_partition
        self._indicators = AttrDict(
            {
                field: [
                    [
                        ffunc.Function(fs, name=f"{field}_error_indicator")
                        for j in range(P.num_exports_per_subinterval[i] - 1)
                    ]
                    for i, fs in enumerate(P0_spaces)
                ]
                for field in P.fields
            }
        )

    @property
    def indicators(self):
        if self._indicators is None:
            self._create_indicators()
        return self._indicators

    def __getitem__(self, key):
        return self.indicators[key]
