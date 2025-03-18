import ufl
from finat.ufl import FiniteElement, TensorElement, VectorElement


class Field:
    """
    A class to represent a field.
    """

    def __init__(self, name, finite_element=None, solved_for=True, unsteady=True):
        """
        Constructs all the necessary attributes for the field object.

        :arg name: The name of the field.
        :type name: :class:`str`
        :arg finite_element: The finite element associated with the field (default is
            Real space on an interval).
        :type finite_element: :class:`~.FiniteElement`
        :arg solved_for: Indicates if the field is to be solved for (default is True).
        :type solved_for: :class:`bool`
        :arg unsteady: Indicates if the field is time-dependent (default is True).
        :type unsteady: :class:`bool`
        """
        if not isinstance(name, str):
            raise TypeError("Field name must be a string.")
        self.name = name
        if finite_element is None:
            finite_element = FiniteElement("Real", ufl.interval, 0)
        if not isinstance(
            finite_element, (FiniteElement, VectorElement, TensorElement)
        ):
            raise TypeError(
                "Field finite element must be a FiniteElement, VectorElement, or"
                " TensorElement object."
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
