import unittest

from goalie.options import AdaptParameters, GoalOrientedAdaptParameters


# TODO: Subclass the test cases to simplify future extension
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
        msg = (
            "Expected attribute 'element_rtol' to be of type 'float' or 'int', not"
            " 'str'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_drop_out_converged_type_error(self):
        with self.assertRaises(TypeError) as cm:
            AdaptParameters({"drop_out_converged": 0})
        msg = "Expected attribute 'drop_out_converged' to be of type 'bool', not 'int'."
        self.assertEqual(str(cm.exception), msg)


class TestGoalOrientedAdaptParameters(unittest.TestCase):
    """
    Unit tests for the :class:`GoalOrientedAdaptParameters` class.
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
        ap = GoalOrientedAdaptParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_str(self):
        ap = GoalOrientedAdaptParameters()
        self.assertEqual(str(ap), str(self.defaults))

    def test_repr(self):
        ap = GoalOrientedAdaptParameters()
        expected = (
            "GoalOrientedAdaptParameters(qoi_rtol=0.001, estimator_rtol=0.001,"
            " convergence_criteria=any, miniter=3, maxiter=35, element_rtol=0.001,"
            " drop_out_converged=False)"
        )
        self.assertEqual(repr(ap), expected)

    def test_convergence_criteria_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedAdaptParameters({"convergence_criteria": 0})
        msg = (
            "Expected attribute 'convergence_criteria' to be of type 'str', not 'int'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_convergence_criteria_value_error(self):
        with self.assertRaises(ValueError) as cm:
            GoalOrientedAdaptParameters({"convergence_criteria": "both"})
        msg = (
            "Unsupported value 'both' for 'convergence_criteria'. Choose from"
            " ['all', 'any']."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_qoi_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedAdaptParameters({"qoi_rtol": "0.001"})
        msg = "Expected attribute 'qoi_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_estimator_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            GoalOrientedAdaptParameters({"estimator_rtol": "0.001"})
        msg = (
            "Expected attribute 'estimator_rtol' to be of type 'float' or 'int', not"
            " 'str'."
        )
        self.assertEqual(str(cm.exception), msg)


class TestOptimisationParameters(unittest.TestCase):
    """
    Unit tests for the base :class:`~.OptimisationParameters` class.
    """

    def setUp(self):
        self.defaults = {
            "R_space": False,
            "lr": 0.001,
            "line_search": True,
            "lr_min": 1.0e-08,
            "ls_rtol": 0.1,
            "ls_frac": 0.5,
            "ls_maxiter": 100,
            "maxiter": 35,
            "gtol": 1.0e-05,
            "gtol_loose": 1.0e-05,
            "dtol": 1.1,
        }

    def test_defaults(self):
        ap = OptimisationParameters()
        for key, value in self.defaults.items():
            self.assertEqual(ap[key], value)

    def test_repr(self):
        ap = OptimisationParameters()
        expected = (
            "OptimisationParameters(R_space=False, lr=0.001, line_search=True,"
            " lr_min=1e-08, ls_rtol=0.1, ls_frac=0.5, ls_maxiter=100, maxiter=35,"
            " gtol=1e-05, gtol_loose=1e-05, dtol=1.1)"
        )
        print(repr(ap))
        self.assertEqual(repr(ap), expected)

    def test_R_space_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"R_space": 0})
        msg = "Expected attribute 'R_space' to be of type 'bool', not 'int'."
        self.assertEqual(str(cm.exception), msg)

    def test_lr_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"lr": "0.001"})
        msg = "Expected attribute 'lr' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_line_search_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"line_search": 0})
        msg = "Expected attribute 'line_search' to be of type 'bool', not 'int'."
        self.assertEqual(str(cm.exception), msg)

    def test_lr_min_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"lr_min": "1.0e-08"})
        msg = "Expected attribute 'lr_min' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_ls_rtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"ls_rtol": "0.1"})
        msg = "Expected attribute 'ls_rtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_ls_frac_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"ls_frac": "0.5"})
        msg = "Expected attribute 'ls_frac' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_ls_maxiter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"ls_maxiter": 100.0})
        msg = "Expected attribute 'ls_maxiter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_maxiter_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"maxiter": 35.0})
        msg = "Expected attribute 'maxiter' to be of type 'int', not 'float'."
        self.assertEqual(str(cm.exception), msg)

    def test_gtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"gtol": "1.0e-05"})
        msg = "Expected attribute 'gtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)

    def test_gtol_loose_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"gtol_loose": "1.0e-05"})
        msg = (
            "Expected attribute 'gtol_loose' to be of type 'float' or 'int', not 'str'."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_dtol_type_error(self):
        with self.assertRaises(TypeError) as cm:
            OptimisationParameters({"dtol": "1.1"})
        msg = "Expected attribute 'dtol' to be of type 'float' or 'int', not 'str'."
        self.assertEqual(str(cm.exception), msg)


if __name__ == "__main__":
    unittest.main()
