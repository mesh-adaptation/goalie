r"""
Nested dictionaries of solution data :class:`~.Function`\s.
"""

from abc import ABC, abstractmethod

import firedrake.function as ffunc
import firedrake.functionspace as ffs

from .utility import AttrDict

__all__ = [
    "ForwardSolutionData",
    "AdjointSolutionData",
    "IndicatorData",
]


class FunctionData(ABC):
    """
    Abstract base class for classes holding field data.
    """

    @abstractmethod
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
        self.labels = self._label_dict[
            "steady" if time_partition.steady else "unsteady"
        ]

    def _create_data(self):
        assert self._label_dict
        tp = self.time_partition
        self._data = AttrDict(
            {
                field: AttrDict(
                    {
                        label: [
                            [
                                ffunc.Function(fs, name=f"{field}_{label}")
                                for j in range(tp.num_exports_per_subinterval[i] - 1)
                            ]
                            for i, fs in enumerate(self.function_spaces[field])
                        ]
                        for label in self.labels
                    }
                )
                for field in tp.field_names
            }
        )

    @property
    def _data_by_field(self):
        """
        Extract field data array in the default layout: as a doubly-nested dictionary
        whose first key is the field name and second key is the field label. Entries
        of the doubly-nested dictionary are doubly-nested lists, indexed first by
        subinterval and then by export.
        """
        if self._data is None:
            self._create_data()
        return self._data

    def __getitem__(self, key):
        return self._data_by_field[key]

    def items(self):
        return self._data_by_field.items()

    @property
    def _data_by_label(self):
        """
        Extract field data array in an alternative layout: as a doubly-nested dictionary
        whose first key is the field label and second key is the field name. Entries
        of the doubly-nested dictionary are doubly-nested lists, which retain the default
        layout: indexed first by subinterval and then by export.
        """
        tp = self.time_partition
        return AttrDict(
            {
                label: AttrDict(
                    {f: self._data_by_field[f][label] for f in tp.field_names}
                )
                for label in self.labels
            }
        )

    @property
    def _data_by_subinterval(self):
        """
        Extract field data array in an alternative format: as a list indexed by
        subinterval. Entries of the list are doubly-nested dictionaries, which retain
        the default layout: with the first key being field name and the second key being
        the field label. Entries of the doubly-nested dictionaries are lists of field
        data, indexed by export.
        """
        tp = self.time_partition
        return [
            AttrDict(
                {
                    field: AttrDict(
                        {
                            label: self._data_by_field[field][label][subinterval]
                            for label in self.labels
                        }
                    )
                    for field in tp.field_names
                }
            )
            for subinterval in range(tp.num_subintervals)
        ]

    def extract(self, layout="field"):
        """
        Extract field data array in a specified layout.

        The default layout is a doubly-nested dictionary whose first key is the field
        name and second key is the field label. Entries of the doubly-nested dictionary
        are doubly-nested lists, indexed first by subinterval and then by export. That
        is: ``data[field][label][subinterval][export]``.

        Choosing a different layout simply promotes the specified variable to first
        access:
        * ``layout == "label"`` implies ``data[label][field][subinterval][export]``
        * ``layout == "subinterval"`` implies ``data[subinterval][field][label][export]``

        The export index is not promoted because the number of exports may differ across
        subintervals.

        :kwarg layout: the layout to promote, as described above
        :type layout: :class:`str`
        """
        if layout == "field":
            return self._data_by_field
        elif layout == "label":
            return self._data_by_label
        elif layout == "subinterval":
            return self._data_by_subinterval
        else:
            raise ValueError(f"Layout type '{layout}' not recognised.")


class ForwardSolutionData(FunctionData):
    """
    Class representing solution data for general forward problems.

    For a given exported timestep, the field types are:

    * ``'forward'``: the forward solution after taking the timestep;
    * ``'forward_old'``: the forward solution before taking the timestep (provided
      the problem is not steady-state).
    """

    def __init__(self, *args, **kwargs):
        self._label_dict = {
            "steady": ("forward",),
            "unsteady": ("forward", "forward_old"),
        }
        super().__init__(*args, **kwargs)


class AdjointSolutionData(FunctionData):
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
        self._label_dict = {
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
        self._label_dict = {
            time_dep: ("error_indicator",) for time_dep in ("steady", "unsteady")
        }
        super().__init__(
            time_partition,
            {
                key: [ffs.FunctionSpace(mesh, "DG", 0) for mesh in meshes]
                for key in time_partition.field_names
            },
        )

    @property
    def _data_by_field(self):
        """
        Extract indicator data array in the default layout: as a dictionary keyed with
        the field name. Entries of the dictionary are doubly-nested lists, indexed first
        by subinterval and then by export.
        """
        if self._data is None:
            self._create_data()
        return AttrDict(
            {
                field: self._data[field]["error_indicator"]
                for field in self.time_partition.field_names
            }
        )

    @property
    def _data_by_label(self):
        """
        For indicator data there is only one field label (``"error_indicator"``), so
        this method just delegates to :meth:`~._data_by_field`.
        """
        return self._data_by_field

    @property
    def _data_by_subinterval(self):
        """
        Extract indicator data array in an alternative format: as a list indexed by
        subinterval. Entries of the list are dictionaries, keyed by field label.
        Entries of the dictionaries are lists of field data, indexed by export.
        """
        tp = self.time_partition
        return [
            AttrDict({f: self._data_by_field[f][subinterval] for f in tp.field_names})
            for subinterval in range(tp.num_subintervals)
        ]
