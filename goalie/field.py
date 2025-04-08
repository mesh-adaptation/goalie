import ufl
from finat.ufl import (
    FiniteElement,
    FiniteElementBase,
    VectorElement,
)
from firedrake.functionspace import FunctionSpace, make_scalar_element


class Field:
    """
    A class to represent a field.
    """

    def __init__(
        self,
        name,
        finite_element=None,
        vector=None,
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

        To account for mixed and tensor elements, please fully specify the element and
        pass it via the `finite_element` keyword argument.

        :arg name: The name of the field.
        :type name: :class:`str`
        :kwarg finite_element: The finite element associated with the field (default is
            Real space on an interval).
        :type finite_element: :class:`~.FiniteElement`
        :kwarg vector: Is the element a vector element? (default is False)
        :type vector: :class:`bool`
        :arg solved_for: Indicates if the field is to be solved for (default is True).
        :type solved_for: :class:`bool`
        :arg unsteady: Indicates if the field is time-dependent (default is True).
        :type unsteady: :class:`bool`
        """
        assert isinstance(name, str), "Field name must be a string."
        self.name = name
        if finite_element is not None:
            if not isinstance(finite_element, FiniteElementBase):
                raise TypeError(
                    "Field finite element must be a FiniteElement, MixedElement,"
                    " VectorElement, or TensorElement object."
                )
            if vector is not None:
                raise ValueError(
                    "The finite_element and vector arguments cannot be used in"
                    " conjunction."
            )
        self.finite_element = finite_element
        self.vector = False if vector is None else vector
        self.family = kwargs.pop("family", None)
        self.degree = kwargs.pop("degree", None)
        self.vfamily = kwargs.pop("vfamily", None)
        self.vdegree = kwargs.pop("vdegree", None)
        self.variant = kwargs.pop("variant", None)
        if kwargs:
            raise ValueError(f"Unexpected keyword argument '{list(kwargs.keys())[0]}'.")
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

    def get_element(self, mesh):
        """
        Given a mesh, return the finite element associated with the field.

        :arg mesh: The mesh to use for the finite element.
        :type mesh: :class:`~.Mesh`
        :return: The finite element associated with the field.
        :rtype: An appropriate subclass of :class:`~.FiniteElementBase`
        """
        if self.finite_element is not None:
            assert self.finite_element.cell == mesh.coordinates.ufl_element().cell
            return self.finite_element

        if self.family is not None:
            finite_element = make_scalar_element(
                mesh,
                self.family,
                self.degree,
                self.vfamily,
                self.vdegree,
                self.variant,
            )

            if self.vector:
                finite_element = VectorElement(
                    finite_element, dim=finite_element.cell.topological_dimension()
                )
            return finite_element

        return FiniteElement("Real", ufl.interval, 0)

    def get_function_space(self, mesh):
        """
        Given a mesh, return the function space associated with the field.

        :arg mesh: The mesh to use for the function space.
        :type mesh: :class:`~.Mesh`
        :return: The function space associated with the field.
        :rtype: :class:`~firedrake.functionspaceimpl.FunctionSpace`
        """
        finite_element = self.get_element(mesh)
        return FunctionSpace(mesh, finite_element)
