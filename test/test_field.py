import unittest

import ufl
from finat.ufl import FiniteElement
from firedrake.utility_meshes import UnitIntervalMesh, UnitTriangleMesh

from goalie.field import Field


def p1_element():
    return FiniteElement("Lagrange", ufl.triangle, 1)


def real_element():
    return FiniteElement("Real", ufl.interval, 0)


def mesh1d():
    return UnitIntervalMesh(1)


def mesh2d():
    return UnitTriangleMesh()


class TestExceptions(unittest.TestCase):
    """
    Test exceptions raised by Field class.

    NOTE: We don't check the exact exception raised in the
    test_make_scalar_element_error* tests because those errors would be raised in
    Firedrake.
    """

    def test_make_scalar_element_error1(self):
        with self.assertRaises(AttributeError):
            Field("field", mesh="mesh", family="Real")

    def test_make_scalar_element_error2(self):
        with self.assertRaises(ValueError):
            Field("field", mesh=mesh1d(), family="family")

    def test_make_scalar_element_error3(self):
        with self.assertRaises(ValueError):
            Field("field", mesh=mesh1d(), family="Real", degree=-1)

    def test_rank_notimplementederror(self):
        with self.assertRaises(NotImplementedError) as cm:
            Field("field", rank=2)
        msg = (
            "rank=2 not supported. Please fully specify your element using the"
            " finite_element argument instead."
        )
        self.assertEqual(str(cm.exception), msg)

    def test_element_and_rank_error(self):
        with self.assertRaises(Exception) as cm:
            Field("field", p1_element(), rank=0)
        msg = "The finite_element and rank arguments cannot be used in conjunction."
        self.assertEqual(str(cm.exception), msg)

    def test_field_invalid_finite_element(self):
        with self.assertRaises(TypeError) as cm:
            Field("field", "element")
        msg = (
            "Field finite element must be a FiniteElement, MixedElement, VectorElement,"
            " or TensorElement object."
        )
        self.assertEqual(str(cm.exception), msg)


class TestInit(unittest.TestCase):
    """Test initialisation of Field class."""

    def test_field_defaults(self):
        field = Field("field", p1_element())
        self.assertTrue(field.solved_for)
        self.assertTrue(field.unsteady)

    def test_field_initialization(self):
        field = Field(
            name="field",
            finite_element=p1_element(),
            solved_for=False,
            unsteady=False,
        )
        self.assertEqual(field.name, "field")
        self.assertEqual(field.finite_element, p1_element())
        self.assertFalse(field.solved_for)
        self.assertFalse(field.unsteady)

    def test_field_alternative_real(self):
        field = Field(
            name="field",
            mesh=mesh1d(),
            family="Real",
            degree=0,
        )
        self.assertEqual(field.finite_element, real_element())

    def test_field_alternative_p1(self):
        field = Field(
            name="field",
            mesh=mesh2d(),
            family="Lagrange",
            degree=1,
        )
        self.assertEqual(field.finite_element, p1_element())


class TestInterrogation(unittest.TestCase):
    """Test interrogation of Field class."""

    def test_str(self):
        self.assertEqual(str(Field("field", p1_element())), "Field(field)")

    def test_repr(self):
        expected_repr = (
            "Field('field', <CG1 on a triangle>, solved_for=True, unsteady=True)"
        )
        self.assertEqual(repr(Field("field", p1_element())), expected_repr)

    def test_eq(self):
        field1 = Field("field", p1_element(), solved_for=True, unsteady=True)
        field2 = Field("field", p1_element(), solved_for=True, unsteady=True)
        self.assertEqual(field1, field2)

    def test_ne_name(self):
        field1 = Field("field1", p1_element(), solved_for=True, unsteady=True)
        field2 = Field("field2", p1_element(), solved_for=True, unsteady=True)
        self.assertNotEqual(field1, field2)

    def test_ne_element(self):
        p2_element = FiniteElement("Lagrange", ufl.triangle, 2)
        field1 = Field("field", p1_element(), solved_for=True, unsteady=True)
        field2 = Field("field", p2_element, solved_for=True, unsteady=True)
        self.assertNotEqual(field1, field2)

    def test_ne_solved_for(self):
        field1 = Field("field", p1_element(), solved_for=True, unsteady=True)
        field2 = Field("field", p1_element(), solved_for=False, unsteady=True)
        self.assertNotEqual(field1, field2)

    def test_ne_unsteady(self):
        field1 = Field("field", p1_element(), solved_for=True, unsteady=True)
        field2 = Field("field", p1_element(), solved_for=True, unsteady=False)
        self.assertNotEqual(field1, field2)


if __name__ == "__main__":
    unittest.main()
