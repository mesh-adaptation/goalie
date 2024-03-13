"""
Partitioning for the temporal domain.
"""
from .log import debug
from collections.abc import Iterable
import numpy as np


__all__ = ["TimePartition", "TimeInterval", "TimeInstant"]


class TimePartition:
    """
    A partition of the time interval of interest into subintervals.

    The subintervals are assumed to be uniform in length. However, different timestep
    values may be used on each subinterval.
    """

    def __init__(
        self,
        end_time,
        num_subintervals,
        timesteps,
        fields,
        num_timesteps_per_export=1,
        start_time=0.0,
        subintervals=None,
        field_types=None,
    ):
        r"""
        :arg end_time: end time of the interval of interest
        :type end_time: :class:`float` or :class:`int`
        :arg num_subintervals: number of subintervals in the partition
        :type num_subintervals: :class:`int`
        :arg timesteps: a list timesteps to be used on each subinterval, or a single
            timestep to use for all subintervals
        :type timesteps: :class:`list` of :class:`float`\s or :class:`float`
        :arg fields: the list of field names to consider
        :type fields: :class:`list` of :class:`str`\s or :class:`str`
        :kwarg num_timesteps_per_export: a list of numbers of timesteps per export for
            each subinterval, or a single number to use for all subintervals
        :type num_timesteps_per_export: :class:`list` of :class`int`\s or :class:`int`
        :kwarg start_time: start time of the interval of interest
        :type start_time: :class:`float` or :class:`int`
        :kwarg subinterals: sequence of subintervals (which need not be of uniform
            length), or ``None`` to use uniform subintervals (the default)
        :type subintervals: :class:`list` of :class:`tuple`\s
        :kwarg field_types: a list of strings indicating whether each field is
            'unsteady' or 'steady', i.e., does the corresponding equation involve time
            derivatives or not?
        :type field_types: :class:`list` of :class:`str`\s or :class:`str`
        """
        debug(100 * "-")
        if isinstance(fields, str):
            fields = [fields]
        self.fields = fields
        self.start_time = start_time
        self.end_time = end_time
        self.num_subintervals = int(np.round(num_subintervals))
        if not np.isclose(num_subintervals, self.num_subintervals):
            raise ValueError(
                f"Non-integer number of subintervals '{num_subintervals}'."
            )
        self.debug("num_subintervals")
        self.interval = (self.start_time, self.end_time)
        self.debug("interval")

        # Get subintervals
        self.subintervals = subintervals
        if self.subintervals is None:
            subinterval_time = (self.end_time - self.start_time) / num_subintervals
            self.subintervals = [
                (
                    self.start_time + i * subinterval_time,
                    self.start_time + (i + 1) * subinterval_time,
                )
                for i in range(num_subintervals)
            ]
        self._check_subintervals()
        self.debug("subintervals")

        # Get timestep on each subinterval
        if not isinstance(timesteps, Iterable):
            timesteps = [timesteps] * len(self)
        self.timesteps = timesteps
        self._check_timesteps()
        self.debug("timesteps")

        # Get number of timesteps on each subinterval
        self.num_timesteps_per_subinterval = []
        for i, ((ts, tf), dt) in enumerate(zip(self.subintervals, self.timesteps)):
            num_timesteps = (tf - ts) / dt
            self.num_timesteps_per_subinterval.append(int(np.round(num_timesteps)))
            if not np.isclose(num_timesteps, self.num_timesteps_per_subinterval[-1]):
                raise ValueError(
                    f"Non-integer number of timesteps on subinterval {i}:"
                    f" {num_timesteps}."
                )
        self.debug("num_timesteps_per_subinterval")

        # Get num timesteps per export
        if not isinstance(num_timesteps_per_export, Iterable):
            num_timesteps_per_export = [num_timesteps_per_export] * len(self)
        self.num_timesteps_per_export = num_timesteps_per_export
        self._check_num_timesteps_per_export()
        self.debug("num_timesteps_per_export")

        # Get num exports per subinterval
        self.num_exports_per_subinterval = [
            tsps // tspe + 1
            for tspe, tsps in zip(
                self.num_timesteps_per_export, self.num_timesteps_per_subinterval
            )
        ]
        self.debug("num_exports_per_subinterval")
        self.steady = (
            self.num_subintervals == 1 and self.num_timesteps_per_subinterval[0] == 1
        )
        self.debug("steady")

        # Process field types
        if field_types is None:
            field_types = ["steady" if self.steady else "unsteady"] * len(self.fields)
        elif isinstance(field_types, str):
            field_types = [field_types]
        self.field_types = field_types
        self._check_field_types()
        debug("field_types")
        debug(100 * "-")

    def debug(self, attr):
        """
        Print attribute 'msg' for debugging purposes.

        :arg attr: the attribute to display debugging information for
        """
        try:
            val = self.__getattribute__(attr)
        except AttributeError:
            raise AttributeError(
                f"Attribute '{attr}' cannot be debugged because it doesn't exist."
            )
        label = " ".join(attr.split("_"))
        debug(f"TimePartition: {label:25s} {val}")

    def __str__(self):
        return f"{self.subintervals}"

    def __repr__(self):
        timesteps = ", ".join([str(dt) for dt in self.timesteps])
        fields = ", ".join([f"'{field}'" for field in self.fields])
        return (
            f"TimePartition("
            f"end_time={self.end_time}, "
            f"num_subintervals={self.num_subintervals}, "
            f"timesteps=[{timesteps}], "
            f"fields=[{fields}])"
        )

    def __len__(self):
        return self.num_subintervals

    def __getitem__(self, index_or_slice):
        """
        :arg index_or_slice: an index or slice to generate a sub-time partition for
        :type index_or_slice: :class:`int` or :class:`slice`
        :returns: a time partition for the given index or slice
        :rtype: :class:`~.TimePartition`
        """
        sl = index_or_slice
        if not isinstance(sl, slice):
            sl = slice(sl, sl + 1, 1)
        step = sl.step or 1
        if step != 1:
            raise NotImplementedError(
                "Can only currently handle slices with step size 1."
            )
        num_subintervals = len(range(sl.start, sl.stop, step))
        return TimePartition(
            end_time=self.subintervals[sl.stop - 1][1],
            num_subintervals=num_subintervals,
            timesteps=self.timesteps[sl],
            fields=self.fields,
            num_timesteps_per_export=self.num_timesteps_per_export[sl],
            start_time=self.subintervals[sl.start][0],
            field_types=self.field_types,
        )

    @property
    def num_timesteps(self):
        """
        :returns the total number of timesteps
        :rtype: :class:`int`
        """
        return sum(self.num_timesteps_per_subinterval)

    def _check_subintervals(self):
        if len(self.subintervals) != self.num_subintervals:
            raise ValueError(
                "Number of subintervals provided differs from num_subintervals:"
                f" {len(self.subintervals)} != {self.num_subintervals}."
            )
        if not np.isclose(self.subintervals[0][0], self.start_time):
            raise ValueError(
                "The first subinterval does not start at the start time:"
                f" {self.subintervals[0][0]} != {self.start_time}."
            )
        for i in range(self.num_subintervals - 1):
            if not np.isclose(self.subintervals[i][1], self.subintervals[i + 1][0]):
                raise ValueError(
                    f"The end of subinterval {i} does not match the start of"
                    f" subinterval {i+1}: {self.subintervals[i][1]} !="
                    f" {self.subintervals[i+1][0]}."
                )
        if not np.isclose(self.subintervals[-1][1], self.end_time):
            raise ValueError(
                "The final subinterval does not end at the end time:"
                f" {self.subintervals[-1][1]} != {self.end_time}."
            )

    def _check_timesteps(self):
        if len(self.timesteps) != self.num_subintervals:
            raise ValueError(
                "Number of timesteps does not match num_subintervals:"
                f" {len(self.timesteps)} != {self.num_subintervals}."
            )

    def _check_num_timesteps_per_export(self):
        if len(self.num_timesteps_per_export) != len(
            self.num_timesteps_per_subinterval
        ):
            raise ValueError(
                "Number of timesteps per export and subinterval do not match:"
                f" {len(self.num_timesteps_per_export)}"
                f" != {len(self.num_timesteps_per_subinterval)}."
            )
        for i, (tspe, tsps) in enumerate(
            zip(self.num_timesteps_per_export, self.num_timesteps_per_subinterval)
        ):
            if not isinstance(tspe, int):
                raise TypeError(
                    f"Expected number of timesteps per export on subinterval {i} to be"
                    f" an integer, not '{type(tspe)}'."
                )
            if tsps % tspe != 0:
                raise ValueError(
                    "Number of timesteps per export does not divide number of"
                    f" timesteps per subinterval on subinterval {i}:"
                    f" {tsps} | {tspe} != 0."
                )

    def _check_field_types(self):
        if len(self.fields) != len(self.field_types):
            raise ValueError(
                "Number of fields does not match number of field types:"
                f" {len(self.fields)} != {len(self.field_types)}."
            )
        for field, field_type in zip(self.fields, self.field_types):
            if field_type not in ("unsteady", "steady"):
                raise ValueError(
                    f"Expected field type for field '{field}' to be either 'unsteady'"
                    f" or 'steady', but got '{field_type}'."
                )

    def __eq__(self, other):
        if len(self) != len(other):
            return False
        return (
            np.allclose(self.subintervals, other.subintervals)
            and np.allclose(self.timesteps, other.timesteps)
            and np.allclose(
                self.num_exports_per_subinterval, other.num_exports_per_subinterval
            )
            and self.fields == other.fields
            and self.field_types == other.field_types
        )

    def __ne__(self, other):
        if len(self) != len(other):
            return True
        return (
            not np.allclose(self.subintervals, other.subintervals)
            or not np.allclose(self.timesteps, other.timesteps)
            or not np.allclose(
                self.num_exports_per_subinterval, other.num_exports_per_subinterval
            )
            or not self.fields == other.fields
            or not self.field_types == other.field_types
        )


class TimeInterval(TimePartition):
    """
    A trivial :class:`~.TimePartition` with a single subinterval.
    """

    def __init__(self, *args, **kwargs):
        if isinstance(args[0], tuple):
            assert len(args[0]) == 2
            kwargs["start_time"] = args[0][0]
            end_time = args[0][1]
        else:
            end_time = args[0]
        timestep = args[1]
        fields = args[2]
        super().__init__(end_time, 1, timestep, fields, **kwargs)

    def __repr__(self):
        return (
            f"TimeInterval("
            f"end_time={self.end_time}, "
            f"timestep={self.timestep}, "
            f"fields={self.fields})"
        )

    @property
    def timestep(self):
        """
        :returns: the timestep used on the single interval
        :rtype: :class:`float`
        """
        return self.timesteps[0]


class TimeInstant(TimeInterval):
    """
    A :class:`~.TimePartition` for steady-state problems.

    Under the hood this means dividing :math:`[0,1)` into a single timestep.
    """

    def __init__(self, fields, **kwargs):
        if "end_time" in kwargs:
            if "time" in kwargs:
                raise ValueError("Both 'time' and 'end_time' are set.")
            time = kwargs.pop("end_time")
        else:
            time = kwargs.pop("time", 1.0)
        timestep = time
        super().__init__(time, timestep, fields, **kwargs)

    def __str__(self):
        return f"({self.end_time})"

    def __repr__(self):
        return f"TimeInstant(" f"time={self.end_time}, " f"fields={self.fields})"
