import unittest

import ufl
from finat.ufl import FiniteElement

from goalie.field import Field


def p1_element():
    return FiniteElement("Lagrange", ufl.triangle, 1)


class TestExceptions(unittest.TestCase):
    """Test exceptions raised by Field class."""

    def test_field_invalid_name(self):
        with self.assertRaises(TypeError) as cm:
            Field(123, p1_element())
        self.assertEqual(str(cm.exception), "Field name must be a string.")

    def test_field_invalid_finite_element(self):
        with self.assertRaises(TypeError) as cm:
            Field("field", "element")
        msg = (
            "Field finite element must be a FiniteElement, VectorElement, or"
            " TensorElement object."
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
