class Solver:
    r"""
    Base class for all solvers.

    Your solver should inherit from this class and implement the following methods:

    - :meth:`get_function_spaces`
    - :meth:`get_solver`.

    Additionally, if your problem requires non-zero initial conditions (for
    time-dependent problems) or a non-zero initial guess (for time-independent
    problems), you should implement the :meth:`get_initial_condition` method.

    If your solver is solving an adjoint problem, you should implement the
    :meth:`get_qoi` method.
    """

    def get_function_spaces(self, *args, **kwargs):
        """
        Construct the function spaces corresponding to each field, for a given mesh.

        Should be overridden by all subclasses.

        Signature for the function:
        ```
            :arg mesh: the mesh to base the function spaces on
            :type mesh: :class:`firedrake.mesh.MeshGeometry`
            :returns: a dictionary whose keys are field names and whose values are the
                corresponding function spaces
            :rtype: :class:`dict` with :class:`str` keys and
                :class:`firedrake.functionspaceimpl.FunctionSpace` values
        ```
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_function_spaces'"
            " method."
        )

    def get_initial_condition(self, *args, **kwargs):
        """
        Get the initial conditions applied on the first mesh in the sequence.

        :returns: the dictionary, whose keys are field names and whose values are the
            corresponding initial conditions applied
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_initial_condition'"
            " method."
        )

    def get_solver(self, *args, **kwargs):
        """
        Get the function mapping a subinterval index and an initial condition dictionary
        to a dictionary of solutions for the corresponding solver step.

        Should be overridden by all subclasses.

        Signature for the function:
        ```
            :arg mesh_seq: the sequence of meshes on which to solve the problem
            :type mesh_seq: :class:`~goalie.mesh_seq.MeshSequence`
            :arg index: the subinterval index
            :type index: :class:`int`
            :arg ic: map from fields to the corresponding initial condition components
            :type ic: :class:`dict` with :class:`str` keys and
                :class:`firedrake.function.Function` values
            :return: map from fields to the corresponding solutions
            :rtype: :class:`dict` with :class:`str` keys and
                :class:`firedrake.function.Function` values
        ```
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_solver' method."
        )

    def get_qoi(self, *args, **kwargs):
        """
        Get the function for evaluating the QoI, which has either zero or one arguments,
        corresponding to either an end time or time integrated quantity of interest,
        respectively. If the QoI has an argument then it is for the current time.

        Should be overridden by all subclasses solving adjoint problems.

        Signature for the function to be returned:
        ```
        :arg t: the current time (for time-integrated QoIs)
        :type t: :class:`float`
        :return: the QoI as a 0-form
        :rtype: :class:`ufl.form.Form`
        ```
        """
        raise NotImplementedError(
            f"Solver {self.__class__.__name__} is missing the 'get_qoi' method."
        )
