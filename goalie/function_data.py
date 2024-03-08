r"""
Nested dictionaries of solution data :class:`~.Function`\s.
"""
import firedrake.function as ffunc
import firedrake.functionspace as ffs
from .utility import AttrDict
import abc

__all__ = [
    "ForwardSolutionData",
    "AdjointSolutionData",
    "IndicatorData",
]


class FunctionData(abc.ABC):
    """
    Abstract base class for classes holding field data.
    """

    labels = {}

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

    def _create_data(self):
        assert self.labels
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
                        for label in self.labels[field_type]
                    }
                )
                for field, field_type in zip(P.fields, P.field_types)
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


class ForwardSolutionData(SolutionData):
    """
    Class representing solution data for general forward problems.

    For a given exported timestep, the field types are:

    * ``'forward'``: the forward solution after taking the timestep;
    * ``'forward_old'``: the forward solution before taking the timestep (provided
      the problem is not steady-state).
    """

    def __init__(self, *args, **kwargs):
        self.labels = {"steady": ("forward",), "unsteady": ("forward", "forward_old")}
        super().__init__(*args, **kwargs)


class AdjointSolutionData(SolutionData):
    """
    Class representing solution data for general adjoint problems.

    For a given exported timestep, the field types are:

    * ``'forward'``: the forward solution after taking the timestep;
    * ``'forward_old'``: the forward solution before taking the timestep (provided
      the problem is not steady-state)
    * ``'adjoint'``: the adjoint solution after taking the timestep;
    * ``'adjoint_next'``: the adjoint solution before taking the timestep
      backwards (provided the problem is not steady-state).
    """

    def __init__(self, *args, **kwargs):
        self.labels = {
            "steady": ("forward", "adjoint"),
            "unsteady": ("forward", "forward_old", "adjoint", "adjoint_next"),
        }
        super().__init__(*args, **kwargs)


class IndicatorData(FunctionData):
    """
    Class representing error indicator data.

    Note that this class has a single dictionary with the field name as the key, rather
    than a doubly-nested dictionary.
    """

    def __init__(self, time_partition, meshes):
        """
        :arg time_partition: the :class:`~.TimePartition` used to discretise the problem
            in time
        :arg meshes: the list of meshes used to discretise the problem in space
        """
        self.labels = {
            field_type: ("error_indicator",) for field_type in ("steady", "unsteady")
        }
        P0_spaces = [ffs.FunctionSpace(mesh, "DG", 0) for mesh in meshes]
        super().__init__(
            time_partition, {key: P0_spaces for key in time_partition.fields}
        )

    def _create_data(self):
        assert all(len(labels) == 1 for labels in self.labels.values())
        super()._create_data()
        P = self.time_partition
        self._data = AttrDict(
            {
                field: self.data[field][self.labels[field_type][0]]
                for field, field_type in zip(P.fields, P.field_types)
            }
        )

    @property
    def indicators(self):
        return self.data
