from .utility import AttrDict

__all__ = ["AdaptParameters", "GoalOrientedAdaptParameters", "OptimisationParameters"]


class AdaptParameters(AttrDict):
    """
    A class for holding parameters associated with adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters=None):
        """
        :kwarg parameters: parameters to set
        :type parameters: :class:`dict` with :class:`str` keys and values which may take
            various types
        """
        parameters = parameters or {}

        self["miniter"] = 3  # Minimum iteration count
        self["maxiter"] = 35  # Maximum iteration count
        self["element_rtol"] = 0.001  # Relative tolerance for element count
        self["drop_out_converged"] = False  # Drop out converged subintervals?

        if not isinstance(parameters, dict):
            raise TypeError(
                "Expected 'parameters' keyword argument to be a dictionary, not of"
                f" type '{parameters.__class__.__name__}'."
            )
        for key in parameters:
            if key not in self:
                raise AttributeError(
                    f"{self.__class__.__name__} does not have '{key}' attribute."
                )
        super().__init__(parameters)
        self._check_type("miniter", int)
        self._check_type("maxiter", int)
        self._check_type("element_rtol", (float, int))
        self._check_type("drop_out_converged", bool)

    def _check_type(self, key, expected):
        """
        Check that a given parameter is of the expected type.

        :arg key: the parameter label
        :type key: :class:`str`
        :arg expected: the expected type
        :type expected: :class:`type`
        """
        if not isinstance(self[key], expected):
            if isinstance(expected, tuple):
                name = "' or '".join([e.__name__ for e in expected])
            else:
                name = expected.__name__
            raise TypeError(
                f"Expected attribute '{key}' to be of type '{name}', not"
                f" '{type(self[key]).__name__}'."
            )

    def _check_value(self, key, possibilities):
        """
        Check that a given parameter takes one of the possible values.

        :arg key: the parameter label
        :type key: :class:`str`
        :arg possibilities: all possible values for the parameter
        :type possibilities: :class:`list`
        """
        value = self[key]
        if value not in possibilities:
            raise ValueError(
                f"Unsupported value '{value}' for '{key}'. Choose from {possibilities}."
            )

    def __str__(self):
        return str(dict(self.items()))

    def __repr__(self):
        d = ", ".join([f"{key}={value}" for key, value in self.items()])
        return f"{type(self).__name__}({d})"


class GoalOrientedAdaptParameters(AdaptParameters):
    """
    A class for holding parameters associated with
    goal-oriented adaptive mesh fixed point iteration
    loops.
    """

    def __init__(self, parameters=None):
        """
        :kwarg parameters: parameters to set
        :type parameters: :class:`dict` with :class:`str` keys and values which may take
            various types
        """
        parameters = parameters or {}

        self["qoi_rtol"] = 0.001  # Relative tolerance for QoI
        self["estimator_rtol"] = 0.001  # Relative tolerance for estimator
        self["convergence_criteria"] = "any"  # Mode for convergence checking

        super().__init__(parameters=parameters)

        self._check_type("qoi_rtol", (float, int))
        self._check_type("estimator_rtol", (float, int))
        self._check_type("convergence_criteria", str)
        self._check_value("convergence_criteria", ["all", "any"])


class OptimisationParameters(AttrDict):
    """
    A class for holding parameters associated with PDE-constrained optimisation.
    """

    def __init__(self, parameters=None):
        """
        :kwarg parameters: parameters to set
        :type parameters: :class:`dict` with :class:`str` keys and values which may take
            various types
        """
        parameters = parameters or {}

        self["R_space"] = False  # Is the control variable defined in R-space?
        self["disp"] = 0  # Level of verbosity

        # Parameters for step length and line search
        self["lr"] = 0.001  # Learning rate / step length
        self["lr_min"] = 1.0e-08  # Minimum learning rate
        self["line_search"] = True  # Toggle whether line search should be used
        self["ls_rtol"] = 0.1  # Relative tolerance for line search
        self["ls_frac"] = 0.5  # Fraction to reduce the step by in line search
        self["ls_maxiter"] = 100  # Maximum iteration count for line search

        # Parameters for optimisation routine
        self["maxiter"] = 35  # Maximum iteration count
        self["gtol"] = 1.0e-05  # Relative tolerance for gradient
        self["gtol_loose"] = 1.0e-05  # TODO: Explanation
        self["dtol"] = 1.1  # Divergence tolerance

        super().__init__(parameters=parameters)

        self._check_type("Rspace", bool)
        self._check_type("disp", int)
        self._check_type("lr", (float, int))
        self._check_type("lr_min", (float, int))
        self._check_type("line_search", bool)
        self._check_type("ls_rtol", (float, int))
        self._check_type("ls_frac", (float, int))
        self._check_type("ls_maxiter", int)
        self._check_type("maxiter", int)
        self._check_type("gtol", (float, int))
        self._check_type("gtol_loose", (float, int))
        self._check_type("dtol", (float, int))
