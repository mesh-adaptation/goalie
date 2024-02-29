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

    _label_dict = {}

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
                        for label in self._label_dict[field_type]
                    }
                )
                for field, field_type in zip(tp.fields, tp.field_types)
            }
        )

    @property
    def data_by_field(self):
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
        return self.data_by_field[key]

    def items(self):
        return self.data_by_field.items()

    @property
    def data_by_label(self):
        """
        Extract field data array in an alternative layout: as a doubly-nested dictionary
        whose first key is the field label and second key is the field name. Entries
        of the doubly-nested dictionary are doubly-nested lists, which retain the default
        layout: indexed first by subinterval and then by export.
        """
        tp = self.time_partition
        labels = self._label_dict["steady"] + self._label_dict["unsteady"]
        return AttrDict(
            {
                label: AttrDict(
                    {field: self.data_by_field[field][label] for field in tp.fields}
                )
                for label in labels
            }
        )

    @property
    def data_by_subinterval(self):
        """
        Extract field data array in an alternative format: as a list indexed by
        subinterval. Entries of the list are doubly-nested dictionaries, which retain
        the default layout: with the first key being field label and the second key being
        the field name. Entries of the doubly-nested dictionaries are lists of field
        data, indexed by export.
        """
        tp = self.time_partition
        labels = self._label_dict["steady"] + self._label_dict["unsteady"]
        return [
            AttrDict(
                {
                    field: AttrDict(
                        {
                            label: self.data_by_field[field][label][subinterval]
                            for label in labels
                        }
                    )
                    for field in tp.fields
                }
            )
            for subinterval in tp.num_subintervals
        ]


class ForwardSolutionData(FunctionData):
    """
    Class representing solution data for general forward problems.
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
            field_type: ("error_indicator",) for field_type in ("steady", "unsteady")
        }
        super().__init__(
            time_partition,
            {
                key: [ffs.FunctionSpace(mesh, "DG", 0) for mesh in meshes]
                for key in time_partition.fields
            },
        )

    @property
    def data_by_field(self):
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
                for field in self.time_partition.fields
            }
        )

    @property
    def data_by_label(self):
        """
        For indicator data there is only one field label (``"error_indicator"``), so
        this method just delegates to :meth:`~.data_by_field`.
        """
        return self.data_by_field

    @property
    def data_by_subinterval(self):
        """
        Extract indicator data array in an alternative format: as a list indexed by
        subinterval. Entries of the list are dictionaries, keyed by field label.
        Entries of the dictionaries are lists of field data, indexed by export.
        """
        tp = self.time_partition
        return [
            AttrDict({self.data_by_field[field][subinterval] for field in tp.fields})
            for subinterval in tp.num_subintervals
        ]
