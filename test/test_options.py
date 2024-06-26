import unittest

from goalie.options import *


class TestAdaptParameters(unittest.TestCase):
    """
    Unit tests for the base :class:`AdaptParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
            "drop_out_converged": False,
        }

    def test_input(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters(1)
        msg = (
            "Expected 'parameters' keyword argument to be a dictionary, not of type"
            " 'int'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_attribute_error(self):
        with self.assertRaises(AttributeError) as cm:
            AdaptParameters({"key": "value"})
        msg = "AdaptParameters does not have 'key' attribute."
        self.assertEqual(str(cm.exception), msg)

    def test_defaults(self):
        ap = AdaptParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_access(self):
        ap = AdaptParameters()
        self.assertEqual(ap.miniter, ap["miniter"])
        self.assertEqual(ap.maxiter, ap["maxiter"])
        self.assertEqual(ap.element_rtol, ap["element_rtol"])
        self.assertEqual(ap.drop_out_converged, ap["drop_out_converged"])

    def test_str(self):
        ap = AdaptParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = AdaptParameters()
        expected = (
            "AdaptParameters(miniter=3, maxiter=35, element_rtol=0.001,"
            " drop_out_converged=False)"
        )
        self.assertEqual(repr(ap), expected)

    def test_miniter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"miniter": 3.0})
        msg = "Expected attribute 'miniter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_maxiter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"maxiter": 35.0})
        msg = "Expected attribute 'maxiter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_element_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"element_rtol": "0.001"})
        msg = "Expected attribute 'element_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_drop_out_converged_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"drop_out_converged": 0})
        msg = "Expected attribute 'drop_out_converged' to be of type 'bool', not 'int'."
        self.assertEqual(str(cm.exception), msg)


class TestGoalOrientedParameters(unittest.TestCase):
    """
    Unit tests for the :class:`GoalOrientedParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "qoi_rtol": 0.001,
            "estimator_rtol": 0.001,
            "convergence_criteria": "any",
            "miniter": 3,
            "maxiter": 35,
            "element_rtol": 0.001,
            "drop_out_converged": False,
        }

    def test_defaults(self):
        ap = GoalOrientedParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_str(self):
        ap = GoalOrientedParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = GoalOrientedParameters()
        expected = (
            "GoalOrientedParameters(qoi_rtol=0.001, estimator_rtol=0.001,"
            " convergence_criteria=any, miniter=3, maxiter=35, element_rtol=0.001,"
            " drop_out_converged=False)"
        )
        self.assertEqual(repr(ap), expected)

    def test_convergence_criteria_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedParameters({"convergence_criteria": 0})
        msg = (
            "Expected attribute 'convergence_criteria' to be of type 'str', not 'int'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_convergence_criteria_value_error(self):
        with self.assertRaises(ValueError) as cm:
            GoalOrientedParameters({"convergence_criteria": "both"})
        msg = "Unsupported value 'both' for 'convergence_criteria'. Choose from ['all', 'any']."
        self.assertEqual(str(cm.exception), msg)

    def test_qoi_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedParameters({"qoi_rtol": "0.001"})
        msg = "Expected attribute 'qoi_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_estimator_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedParameters({"estimator_rtol": "0.001"})
        msg = (
            "Expected attribute 'estimator_rtol' to be of type 'float' or 'int', not"
            " 'str'."
        )
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
