r"""
Nested dictionaries of solution data :class:`~.Function`\s.
"""

from abc import ABC, abstractmethod

import firedrake.function as ffunc
import firedrake.functionspace as ffs
from firedrake import TransferManager
from firedrake.checkpointing import CheckpointFile
from firedrake.output.vtk_output import VTKFile

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
        of the doubly-nested dictionary are doubly-nested lists, which retain the
        default layout: indexed first by subinterval and then by export.
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
        * ``layout == "subinterval"`` implies
          ``data[subinterval][field][label][export]``

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

    def export(self, output_fpath, export_field_types=None, initial_condition=None):
        """
        Export field data to a file. The file format is determined by the extension of
        the output file path. Supported formats are '.pvd' and '.h5'.

        If the output file format is '.pvd', the data is exported as a series of VTK
        files using Firedrake's :class:`~.VTKFile`. Since mixed function spaces are not
        supported by VTK, each subfunction of a mixed function is exported separately.
        If initial conditions are provided and fields other than 'forward' are to be
        exported, the initial values of these other fields are set to 'nan' since they
        are not defined at the initial time (e.g., 'adjoint' fields).

        If the output file format is '.h5', the data is exported as a single HDF5 file
        using Firedrake's :class:`~.CheckpointFile`. If names of meshes in the mesh
        sequence are not unique, they are renamed to ``"mesh_i"``, where ``i`` is the
        subinterval index. Functions are saved with names of the form ``"field_label"``.
        Initial conditions are named in the form ``"field_initial"``. The exported data
        may then be loaded using, for example,

        .. code-block:: python

            with CheckpointFile(output_fpath, "r") as afile:
                first_mesh = afile.load_mesh("mesh_0")
                initial_condition = afile.load_function(first_mesh, "u_initial")
                first_export = afile.load_function(first_mesh, "u_forward", idx=0)

        :arg output_fpath: the path to the output file
        :type output_fpath: :class:`str`
        :kwarg export_field_types: the field types to export; defaults to all available
            field types
        :type export_field_types: :class:`str` or :class:`list` of :class:`str`
        :kwarg initial_condition: if provided, exports the provided initial condition
            for 'forward' fields.
        :type initial_condition: :class:`dict` of :class:`~.Function`
        """
        if export_field_types is None:
            default_export_types = {"forward", "adjoint", "error_indicator"}
            export_field_types = list(set(self.labels) & default_export_types)
        if isinstance(export_field_types, str):
            export_field_types = [export_field_types]
        if not all(field_type in self.labels for field_type in export_field_types):
            raise ValueError(
                f"Field types {export_field_types} not recognised."
                f" Available types are {self.labels}."
            )

        if output_fpath.endswith(".pvd"):
            self._export_vtk(output_fpath, export_field_types, initial_condition)
        elif output_fpath.endswith(".h5"):
            self._export_h5(output_fpath, export_field_types, initial_condition)
        else:
            raise ValueError(
                f"Output file format not recognised: '{output_fpath}'."
                " Supported formats are '.pvd' and '.h5'."
            )

    def _export_vtk(self, output_fpath, export_field_types, initial_condition=None):
        """
        Export field data to a series of VTK files. Arguments are the same as for
        :meth:`~.export`.
        """
        tp = self.time_partition
        outfile = VTKFile(output_fpath, adaptive=True)
        if initial_condition is not None:
            ics = []
            for field, ic in sorted(initial_condition.items()):
                for field_type in export_field_types:
                    icc = ic.copy(deepcopy=True)
                    # If the function space is mixed, rename and append each
                    # subfunction separately
                    if hasattr(ic.function_space(), "num_sub_spaces"):
                        for idx, sf in enumerate(ic.subfunctions):
                            if field_type != "forward":
                                sf = sf.copy(deepcopy=True)
                                sf.assign(float("nan"))
                            sf.rename(f"{field}[{idx}]_{field_type}")
                            ics.append(sf)
                    else:
                        if field_type != "forward":
                            icc.assign(float("nan"))
                        icc.rename(f"{field}_{field_type}")
                        ics.append(icc)
            outfile.write(*ics, time=tp.subintervals[0][0])

        for i in range(tp.num_subintervals):
            for j in range(tp.num_exports_per_subinterval[i] - 1):
                time = (
                    tp.subintervals[i][0]
                    + (j + 1) * tp.timesteps[i] * tp.num_timesteps_per_export[i]
                )
                fs = []
                for field in sorted(tp.field_names):
                    mixed = hasattr(self.function_spaces[field][0], "num_sub_spaces")
                    for field_type in export_field_types:
                        f = self._data[field][field_type][i][j].copy(deepcopy=True)
                        if mixed:
                            for idx, sf in enumerate(f.subfunctions):
                                sf.rename(f"{field}[{idx}]_{field_type}")
                                fs.append(sf)
                        else:
                            f.rename(f"{field}_{field_type}")
                            fs.append(f)
                outfile.write(*fs, time=time)

    def _export_h5(self, output_fpath, export_field_types, initial_condition=None):
        """
        Export field data to an HDF5 file. Arguments are the same as for
        :meth:`~.export`.
        """
        tp = self.time_partition

        # Mesh names must be unique
        mesh_names = [fs.mesh().name for fs in self.function_spaces[tp.field_names[0]]]
        rename_meshes = len(set(mesh_names)) != len(mesh_names)
        with CheckpointFile(output_fpath, "w") as outfile:
            if initial_condition is not None:
                for field, ic in initial_condition.items():
                    outfile.save_function(ic, name=f"{field}_initial")
            for i in range(tp.num_subintervals):
                if rename_meshes:
                    mesh_name = f"mesh_{i}"
                    mesh = self.function_spaces[tp.field_names[0]][i].mesh()
                    mesh.name = mesh_name
                    mesh.topology_dm.name = mesh_name
                for field in tp.field_names:
                    for field_type in export_field_types:
                        name = f"{field}_{field_type}"
                        for j in range(tp.num_exports_per_subinterval[i] - 1):
                            f = self._data[field][field_type][i][j]
                            outfile.save_function(f, name=name, idx=j)

    def transfer(self, target, method="interpolate"):
        """
        Transfer all functions from this :class:`~.FunctionData` object to the target
        :class:`~.FunctionData` object by interpolation, projection or prolongation.

        :arg target: the target :class:`~.FunctionData` object to which to transfer the
            data
        :type target: :class:`~.FunctionData`
        :arg method: the transfer method to use. Either 'interpolate', 'project' or
            'prolong'
        :type method: :class:`str`
        """
        stp = self.time_partition
        ttp = target.time_partition

        if method not in ["interpolate", "project", "prolong"]:
            raise ValueError(
                f"Transfer method '{method}' not supported."
                " Supported methods are 'interpolate', 'project', and 'prolong'."
            )
        if stp.num_subintervals != ttp.num_subintervals:
            raise ValueError(
                "Source and target have different numbers of subintervals."
            )
        if stp.num_exports_per_subinterval != ttp.num_exports_per_subinterval:
            raise ValueError(
                "Source and target have different numbers of exports per subinterval."
            )

        common_fields = set(stp.field_names) & set(ttp.field_names)
        if not common_fields:
            raise ValueError("No common fields between source and target.")

        common_labels = set(self.labels) & set(target.labels)
        if not common_labels:
            raise ValueError("No common labels between source and target.")

        for field in common_fields:
            for label in common_labels:
                for i in range(stp.num_subintervals):
                    for j in range(stp.num_exports_per_subinterval[i] - 1):
                        source_function = self._data[field][label][i][j]
                        target_function = target._data[field][label][i][j]
                        if method == "interpolate":
                            target_function.interpolate(source_function)
                        elif method == "project":
                            target_function.project(source_function)
                        elif method == "prolong":
                            TransferManager().prolong(source_function, target_function)


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
