import ufl
from finat.ufl import (
    FiniteElement,
    FiniteElementBase,
    VectorElement,
)
from firedrake.functionspace import make_scalar_element


class Field:
    """
    A class to represent a field.
    """

    def __init__(
        self,
        name,
        finite_element=None,
        rank=None,
        solved_for=True,
        unsteady=True,
        **kwargs,
    ):
        """
        Constructs all the necessary attributes for the field object.

        The finite element for the Field should be set either using the `finite_element`
        keyword argument or a combination of the `mesh`, `family`, `degree`, `vfamily`,
        `vdegree`, and/or `variant` keyword arguments. For details on these arguments,
        see :class:`firedrake.functionspace.FunctionSpace`.

        If neither are specified, the default finite element is a scalar Real space on
        and interval.

        :arg name: The name of the field.
        :type name: :class:`str`
        :kwarg finite_element: The finite element associated with the field (default is
            Real space on an interval).
        :type finite_element: :class:`~.FiniteElement`
        :kwarg rank: The rank of the field, where 0 implies scalar, 1 implies vector,
            and 2 implies tensor.
        :arg solved_for: Indicates if the field is to be solved for (default is True).
        :type solved_for: :class:`bool`
        :arg unsteady: Indicates if the field is time-dependent (default is True).
        :type unsteady: :class:`bool`
        """
        assert isinstance(name, str), "Field name must be a string."
        self.name = name
        if finite_element is None:
            if kwargs:
                finite_element = make_scalar_element(
                    kwargs.get("mesh"),
                    kwargs.get("family"),
                    kwargs.get("degree"),
                    kwargs.get("vfamily"),
                    kwargs.get("vdegree"),
                    kwargs.get("variant"),
                )
            else:
                finite_element = FiniteElement("Real", ufl.interval, 0)
            if rank is None:
                rank = 0
            if rank == 1:
                finite_element = VectorElement(
                    finite_element, dim=finite_element.cell.topological_dimension()
                )
            elif rank != 0:
                raise NotImplementedError(
                    f"rank={rank} not supported. Please fully specify your element"
                    " using the finite_element argument instead."
                )
        elif rank is not None:
            raise ValueError(
                "The finite_element and rank arguments cannot be used in conjunction."
            )
        if not isinstance(finite_element, FiniteElementBase):
            raise TypeError(
                "Field finite element must be a FiniteElement, MixedElement,"
                " VectorElement, or TensorElement object."
            )
        self.finite_element = finite_element
        assert isinstance(solved_for, bool), "'solved_for' argument must be a bool"
        self.solved_for = solved_for
        assert isinstance(unsteady, bool), "'unsteady' argument must be a bool"
        self.unsteady = unsteady

    def __str__(self):
        return f"Field({self.name})"

    def __repr__(self):
        return (
            f"Field('{self.name}', {self.finite_element}, solved_for={self.solved_for},"
            f" unsteady={self.unsteady})"
        )

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        return (
            self.name == other.name
            and self.finite_element == other.finite_element
            and self.solved_for == other.solved_for
            and self.unsteady == other.unsteady
        )

    def __ne__(self, other):
        return not self.__eq__(other)
