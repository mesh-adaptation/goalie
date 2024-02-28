r"""
Nested dictionaries of solution data :class:`~.Function`\s.
"""
import firedrake.function as ffunc
from .utility import AttrDict
import abc

__all__ = ["ForwardSolutionData"]


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


class ForwardSolutionData(SolutionData):
    """
    Class representing solution data for forward problems.
    """

    labels = ("forward", "forward_old")
