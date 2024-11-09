"""
Testing for the time partition objects.
"""

import unittest

from goalie.field import Field
from goalie.time_partition import TimeInstant, TimeInterval, TimePartition


class BaseTestCase(unittest.TestCase):
    """
    Base class for unit tests related to the time partition objects.
    """

    def setUp(self):
        self.end_time = 1.0
        self.field = Field("field", family="Real")
        self.field_metadata_list = [self.field]
        self.field_metadata_dict = {"field": self.field}


class TestExceptions(BaseTestCase):
    """
    Tests for exceptions raised by the time partition objects.
    """

    def test_time_instant_multiple_kwargs(self):
        with self.assertRaises(ValueError) as cm:
            TimeInstant(self.field, time=1.0, end_time=1.0)
        msg = "Both 'time' and 'end_time' are set."
        self.assertEqual(str(cm.exception), msg)

    def test_noninteger_num_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1.1, 0.5, self.field)
        msg = "Non-integer number of subintervals '1.1'."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_number_of_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(
                1.0, 1, 0.5, self.field, subintervals=[(0.0, 0.5), (0.5, 1.0)]
            )
        msg = "Number of subintervals provided differs from num_subintervals: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_subinterval_start(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, 0.5, self.field, subintervals=[(0.1, 1.0)])
        msg = "The first subinterval does not start at the start time: 0.1 != 0.0."
        self.assertEqual(str(cm.exception), msg)

    def test_inconsistent_subintervals(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(
                1.0, 2, 0.5, self.field, subintervals=[(0.0, 0.6), (0.5, 1.0)]
            )
        msg = (
            "The end of subinterval 0 does not match the start of subinterval 1:"
            " 0.6 != 0.5."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_subinterval_end(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, 0.5, self.field, subintervals=[(0.0, 1.1)])
        msg = "The final subinterval does not end at the end time: 1.1 != 1.0."
        self.assertEqual(str(cm.exception), msg)

    def test_wrong_num_timesteps(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5, 0.5], self.field)
        msg = "Number of timesteps does not match num_subintervals: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_noninteger_num_timesteps_per_subinterval(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.4], self.field)
        msg = "Non-integer number of timesteps on subinterval 0: 2.5."
        self.assertEqual(str(cm.exception), msg)

    def test_noninteger_num_timesteps_per_export(self):
        with self.assertRaises(TypeError) as cm:
            TimePartition(1.0, 1, [0.5], self.field, num_timesteps_per_export=1.1)
        msg = (
            "Expected number of timesteps per export on subinterval 0 to be an integer,"
            " not '<class 'float'>'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_nonmatching_num_timesteps_per_export(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5], self.field, num_timesteps_per_export=[1, 2])
        msg = "Number of timesteps per export and subinterval do not match: 2 != 1."
        self.assertEqual(str(cm.exception), msg)

    def test_indivisible_num_timesteps_per_export(self):
        with self.assertRaises(ValueError) as cm:
            TimePartition(1.0, 1, [0.5], self.field, num_timesteps_per_export=4)
        msg = (
            "Number of timesteps per export does not divide number of timesteps per"
            " subinterval on subinterval 0: 2 | 4 != 0."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_debug_invalid_field(self):
        with self.assertRaises(AttributeError) as cm:
            TimeInstant(self.field).debug("blah")
        msg = "Attribute 'blah' cannot be debugged because it doesn't exist."
        self.assertEqual(str(cm.exception), msg)

    def test_field_typeerror1(self):
        with self.assertRaises(TypeError) as cm:
            TimeInstant("field")
        msg = "field_metadata argument must be a Field or a dict or list thereof."
        self.assertEqual(str(cm.exception), msg)

    def test_field_typeerror2(self):
        with self.assertRaises(TypeError) as cm:
            TimeInstant(["field"])
        self.assertEqual(str(cm.exception), "All fields must be instances of Field.")

    def test_inconsistent_fieldname_valueerror(self):
        with self.assertRaises(ValueError) as cm:
            TimeInstant({"field1": Field("field2", family="Real")})
        msg = "Inconstent field names passed as field_metadata."
        self.assertEqual(str(cm.exception), msg)


class TestSetup(BaseTestCase):
    """
    Tests related to the construction of time partitions.
    """

    def test_time_partition_eq_positive(self):
        time_partition1 = TimePartition(1.0, 1, [1.0], self.field)
        time_partition2 = TimePartition(1.0, 1, [1.0], self.field)
        self.assertTrue(time_partition1 == time_partition2)

    def test_time_partition_eq_negative(self):
        time_partition1 = TimePartition(1.0, 1, [1.0], self.field)
        time_partition2 = TimePartition(2.0, 1, [1.0], self.field)
        self.assertFalse(time_partition1 == time_partition2)

    def test_time_partition_ne_positive(self):
        time_partition1 = TimePartition(1.0, 2, [0.5, 0.5], self.field)
        time_partition2 = TimePartition(1.0, 1, [1.0], self.field)
        self.assertTrue(time_partition1 != time_partition2)

    def test_time_partition_ne_negative(self):
        time_partition1 = TimePartition(
            1.0, 1, [1.0], self.field, num_timesteps_per_export=1
        )
        time_partition2 = TimePartition(1.0, 1, [1.0], self.field)
        self.assertFalse(time_partition1 != time_partition2)

    def test_time_interval_eq_positive(self):
        time_interval1 = TimeInterval(1.0, 1.0, self.field)
        time_interval2 = TimeInterval((0.0, 1.0), 1.0, [self.field])
        self.assertTrue(time_interval1 == time_interval2)

    def test_time_interval_eq_negative(self):
        time_interval1 = TimeInterval(1.0, 1.0, self.field)
        time_interval2 = TimeInterval((0.5, 1.0), 0.5, self.field)
        self.assertFalse(time_interval1 == time_interval2)

    def test_time_interval_ne_positive(self):
        time_interval1 = TimeInterval(1.0, 1.0, self.field)
        time_interval2 = TimeInterval((-0.5, 0.5), 1.0, self.field)
        self.assertTrue(time_interval1 != time_interval2)

    def test_time_interval_ne_negative(self):
        time_interval1 = TimeInterval(1.0, 1.0, self.field)
        time_interval2 = TimeInterval((0.0, 1.0), 1.0, [self.field])
        self.assertFalse(time_interval1 != time_interval2)

    def test_time_instant_eq_positive(self):
        time_instant1 = TimeInstant(self.field, time=1.0)
        time_instant2 = TimeInstant([self.field], time=1.0)
        self.assertTrue(time_instant1 == time_instant2)

    def test_time_instant_eq_negative(self):
        time_instant1 = TimeInstant(self.field, time=1.0)
        time_instant2 = TimeInstant(Field("f", family="Real"), time=1.0)
        self.assertFalse(time_instant1 == time_instant2)

    def test_time_instant_ne_positive(self):
        time_instant1 = TimeInstant(self.field, time=1.0)
        time_instant2 = TimeInstant(self.field, time=2.0)
        self.assertTrue(time_instant1 != time_instant2)

    def test_time_instant_ne_negative(self):
        time_instant1 = TimeInstant(self.field, time=1.0)
        time_instant2 = TimeInstant(self.field, end_time=1.0)
        self.assertFalse(time_instant1 != time_instant2)

    def test_time_partition_eq_interval_positive(self):
        time_partition = TimePartition(1.0, 1, [0.5], [self.field])
        time_interval = TimeInterval(1.0, 0.5, self.field)
        self.assertTrue(time_partition == time_interval)

    def test_time_partition_eq_interval_negative(self):
        time_partition = TimePartition(1.0, 2, [0.5, 0.5], [self.field])
        time_interval = TimeInterval(1.0, 0.5, self.field)
        self.assertFalse(time_partition == time_interval)

    def test_time_partition_ne_interval_positive(self):
        time_partition = TimePartition(0.5, 1, [0.5], self.field)
        time_interval = TimeInterval(1.0, 0.5, self.field)
        self.assertTrue(time_partition != time_interval)

    def test_time_partition_ne_interval_negative(self):
        time_partition = TimePartition(1.0, 1, 0.5, [self.field], start_time=0.0)
        time_interval = TimeInterval(1.0, 0.5, self.field)
        self.assertFalse(time_partition != time_interval)


class TestDropLastSubinterval(unittest.TestCase):
    r"""
    Tests related to :method:\~.drop_last_subinterval` of :class:`~.TimePartition`
    """

    def setUp(self):
        self.end_time = 1.0
        self.field_names = ["field"]

    def get_time_partition(self, n):
        split = self.end_time / n
        timesteps = [split if i % 2 else split / 2 for i in range(n)]
        return TimePartition(self.end_time, n, timesteps, self.field_names)

    def test_drop_last_subinterval(self):
        n = 5
        multi_subintervals = self.get_time_partition(n - 1)
        expected_end_time = [1.0, 0.75, 0.5, 0.25, 0.25]
        expected_interval = [
            (0.0, 1.0),
            (0.0, 0.75),
            (0.0, 0.5),
            (0.0, 0.25),
            (0.0, 0.25),
        ]
        expected_num_subintervals = [4, 3, 2, 1, 1]
        expected_timesteps = [
            [0.125, 0.25, 0.125, 0.25],
            [0.125, 0.25, 0.125],
            [0.125, 0.25],
            [0.125],
            [0.125],
        ]
        expected_subintervals = [
            [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)],
            [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75)],
            [(0.0, 0.25), (0.25, 0.5)],
            [(0.0, 0.25)],
            [(0.0, 0.25)],
        ]
        expected_n_time_per_sub = [[2, 1, 2, 1], [2, 1, 2], [2, 1], [2], [2]]
        expected_n_time_per_xpt = [[1, 1, 1, 1], [1, 1, 1], [1, 1], [1], [1]]
        expected_n_xpt_per_sub = [[3, 2, 3, 2], [3, 2, 3], [3, 2], [3], [3]]
        for i in range(n):
            self.assertEqual(multi_subintervals.end_time, expected_end_time[i])
            self.assertEqual(multi_subintervals.interval, expected_interval[i])
            self.assertEqual(
                multi_subintervals.num_subintervals, expected_num_subintervals[i]
            )
            self.assertEqual(multi_subintervals.timesteps, expected_timesteps[i])
            self.assertEqual(multi_subintervals.subintervals, expected_subintervals[i])
            self.assertEqual(
                multi_subintervals.num_timesteps_per_subinterval,
                expected_n_time_per_sub[i],
            )
            self.assertEqual(
                multi_subintervals.num_timesteps_per_export, expected_n_time_per_xpt[i]
            )
            self.assertEqual(
                multi_subintervals.num_exports_per_subinterval,
                expected_n_xpt_per_sub[i],
            )
            multi_subintervals.drop_last_subinterval()


class TestStringFormatting(unittest.TestCase):
    """
    Test that the :meth:`__str__`` and :meth:`__repr__`` methods work as intended for
    Goalie's time partition objects.
    """

    def get_time_partition(self, n):
        split = self.end_time / n
        timesteps = [split if i % 2 else split / 2 for i in range(n)]
        return TimePartition(self.end_time, n, timesteps, self.field_metadata_list)

    def test_time_partition1_str(self):
        expected = "[(0.0, 1.0)]"
        self.assertEqual(str(self.get_time_partition(1)), expected)

    def test_time_partition2_str(self):
        expected = "[(0.0, 0.5), (0.5, 1.0)]"
        self.assertEqual(str(self.get_time_partition(2)), expected)

    def test_time_partition4_str(self):
        expected = "[(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]"
        self.assertEqual(str(self.get_time_partition(4)), expected)

    def test_time_interval_str(self):
        expected = "[(0.0, 1.0)]"
        time_interval = TimeInterval(self.end_time, [0.5], self.field_metadata_list)
        self.assertEqual(str(time_interval), expected)

    def test_time_instant_str(self):
        expected = "(1.0)"
        time_instant = TimeInstant(self.field_metadata_list, time=self.end_time)
        self.assertEqual(str(time_instant), expected)

    def test_time_partition1_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=1,"
            " timesteps=[0.5], field_metadata=[Field('field', <R? on a unknown cell"
            " type>, solved_for=True, unsteady=True)])"
        )
        self.assertEqual(repr(self.get_time_partition(1)), expected)

    def test_time_partition2_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=2,"
            " timesteps=[0.25, 0.5], field_metadata=[Field('field', <R? on a unknown"
            " cell type>, solved_for=True, unsteady=True)])"
        )
        self.assertEqual(repr(self.get_time_partition(2)), expected)

    def test_time_partition4_repr(self):
        expected = (
            "TimePartition(end_time=1.0, num_subintervals=4,"
            " timesteps=[0.125, 0.25, 0.125, 0.25], field_metadata=[Field('field',"
            " <R? on a unknown cell type>, solved_for=True, unsteady=True)])"
        )
        self.assertEqual(repr(self.get_time_partition(4)), expected)

    def test_time_interval_repr(self):
        expected = (
            "TimeInterval(end_time=1.0, timestep=0.5, field_metadata=[Field('field',"
            " <R? on a unknown cell type>, solved_for=True, unsteady=True)])"
        )
        time_interval = TimeInterval(self.end_time, [0.5], self.field_metadata_list)
        self.assertEqual(repr(time_interval), expected)

    def test_time_instant_repr(self):
        expected = (
            "TimeInstant(time=1.0, field_metadata=[Field('field',"
            " <R? on a unknown cell type>, solved_for=True, unsteady=True)])"
        )
        time_instant = TimeInstant(self.field_metadata_list, time=self.end_time)
        self.assertEqual(repr(time_instant), expected)


class TestIndexing(BaseTestCase):
    r"""
    Unit tests for indexing :class:`~.TimePartition`\s.
    """

    def test_invalid_step(self):
        timesteps = [0.5, 0.25]
        time_partition = TimePartition(
            self.end_time, 2, timesteps, self.field_metadata_list
        )
        with self.assertRaises(NotImplementedError) as cm:
            time_partition[::2]
        msg = "Can only currently handle slices with step size 1."
        self.assertEqual(str(cm.exception), msg)

    def test_time_interval(self):
        time_interval = TimeInterval(self.end_time, [0.5], self.field_metadata_list)
        self.assertEqual(time_interval, time_interval[0])

    def test_time_instant(self):
        time_instant = TimeInstant(self.field_metadata_list, time=self.end_time)
        self.assertEqual(time_instant, time_instant[0])

    def test_time_partition(self):
        timesteps = [0.5, 0.25]
        time_partition = TimePartition(
            self.end_time, 2, timesteps, self.field_metadata_list
        )
        tp0, tp1 = time_partition
        self.assertEqual(len(tp0), 1)
        self.assertEqual(len(tp1), 1)
        self.assertAlmostEqual(tp0.start_time, 0.0)
        self.assertAlmostEqual(tp0.end_time, 0.5)
        self.assertAlmostEqual(tp0.end_time, tp1.start_time)
        self.assertAlmostEqual(tp1.end_time, 1.0)
        self.assertAlmostEqual(tp0.timesteps[0], timesteps[0])
        self.assertAlmostEqual(tp1.timesteps[0], timesteps[1])
        self.assertEqual(tp0.field_metadata, self.field_metadata_dict)
        self.assertEqual(tp0.field_metadata, tp1.field_metadata)


class TestSlicing(BaseTestCase):
    r"""
    Unit tests for slicing :class:`~.TimePartition`\s.
    """

    def test_time_interval(self):
        time_interval = TimeInterval(self.end_time, [0.5], self.field_metadata_list)
        self.assertEqual(time_interval, time_interval[0:1])

    def test_time_instant(self):
        time_instant = TimeInstant(self.field_metadata_list, time=self.end_time)
        self.assertEqual(time_instant, time_instant[0:1])

    def test_time_partition(self):
        timesteps = [0.5, 0.25]
        time_partition = TimePartition(
            self.end_time, 2, timesteps, self.field_metadata_list
        )
        self.assertEqual(time_partition, time_partition[0:2])

    def test_time_partition_slice(self):
        end_time = 0.75
        timesteps = [0.25, 0.05, 0.01]
        time_partition = TimePartition(end_time, 3, timesteps, self.field_metadata_list)
        tp0 = time_partition[0]
        tp12 = time_partition[1:3]
        self.assertEqual(len(tp0), 1)
        self.assertEqual(len(tp12), 2)
        self.assertAlmostEqual(tp0.start_time, 0.0)
        self.assertAlmostEqual(tp0.end_time, 0.25)
        self.assertAlmostEqual(tp0.end_time, tp12.start_time)
        self.assertAlmostEqual(tp12.end_time, end_time)
        self.assertAlmostEqual(tp0.timesteps[0], timesteps[0])
        self.assertAlmostEqual(tp12.timesteps, timesteps[1:3])
        self.assertEqual(tp0.field_metadata, self.field_metadata_dict)
        self.assertEqual(tp0.field_metadata, tp12.field_metadata)
