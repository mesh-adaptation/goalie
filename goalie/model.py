import firedrake
import numpy as np

from .log import DEBUG, debug, logger

__all__ = ["Model"]


class Model:
    # FIXME: Update docstrings
    r"""
    Base class for all models.

    Your model should inherit from this class and implement the following methods:
    - :meth:`get_function_spaces`
    - :meth:`get_solver`.

    Additionally, if your problem requires non-zero initial conditions (for
    time-dependent problems) or a non-zero initial guess (for time-independent
    problems), you should implement the :meth:`get_initial_condition` method.

    If your solver is solving an adjoint problem, you should implement the
    :meth:`get_qoi` method.
    """

    def get_initial_condition(self, time_partition, meshes, fields, function_spaces):
        """
        Get the initial conditions applied on the first mesh in the sequence.

        :returns: the dictionary, whose keys are field names and whose values are the
            corresponding initial conditions applied
        :rtype: :class:`dict` with :class:`str` keys and
            :class:`firedrake.function.Function` values
        """
        return {
            field: firedrake.Function(fs[0]) for field, fs in function_spaces.items()
        }

    def get_solver(self, index, time_partition, meshes, fields, function_spaces):
        # def get_solver(self, *args, **kwargs):
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
            f"Model {self.__class__.__name__} is missing the 'get_solver' method."
        )

    def get_qoi(self, index, time_partition, meshes, fields, function_spaces):
        """
        Get the function for evaluating the QoI, which has either zero or one arguments,
        corresponding to either an end time or time integrated quantity of interest,
        respectively. If the QoI has an argument then it is for the current time.

        Should be overridden by all subclasses.

        Signature for the function to be returned:
        ```
        :arg t: the current time (for time-integrated QoIs)
        :type t: :class:`float`
        :return: the QoI as a 0-form
        :rtype: :class:`ufl.form.Form`
        ```
        """
        raise NotImplementedError(
            f"Model {self.__class__.__name__} is missing the 'get_qoi' method."
        )

    def debug(self, msg):
        """
        Print a ``debug`` message.
        :arg msg: the message to print
        :type msg: :class:`str`
        """
        debug(f"{type(self).__name__}: {msg}")

    def _outputs_consistent(self, time_partition, meshes, fields, function_spaces):
        """
        Assert that function spaces and initial conditions are given in a
        dictionary format with :attr:`Solver.fields` as keys.
        """
        for method in ["initial_condition", "solver"]:
            try:
                method_map = getattr(self, f"get_{method}")
                if method == "initial_condition":
                    method_map = method_map(
                        time_partition, meshes, fields, function_spaces
                    )
                elif method == "solver":
                    # FIXME: _reinitialise_fields is a Solver method... should also move
                    # it inside the if block below
                    # self._reinitialise_fields(self.get_initial_condition())
                    solver_gen = method_map(
                        0, time_partition, meshes, fields, function_spaces
                    )
                    assert hasattr(solver_gen, "__next__"), "get_solver should yield"
                    if logger.level == DEBUG:
                        next(solver_gen)
                        f, f_ = self.fields[next(iter(self.fields))]
                        if np.array_equal(f.vector().array(), f_.vector().array()):
                            self.debug(
                                "Current and lagged solutions are equal. Does the"
                                " solver yield before updating lagged solutions?"
                            )  # noqa
                    break
            except NotImplementedError:
                continue
            assert isinstance(method_map, dict), f"get_{method} should return a dict"
            mesh_seq_fields = set(fields)
            method_fields = set(method_map.keys())
            diff = mesh_seq_fields.difference(method_fields)
            assert len(diff) == 0, f"missing fields {diff} in get_{method}"
            diff = method_fields.difference(mesh_seq_fields)
            assert len(diff) == 0, f"unexpected fields {diff} in get_{method}"
