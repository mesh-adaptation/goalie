import pyadjoint
import pytest


@pytest.fixture(autouse=True)
def handle_taping():
    """
    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    yield
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()


@pytest.fixture(autouse=True, scope="module")
def handle_annotation():
    """
    Since importing firedrake-adjoint modifies a global variable, we need to
    pause annotations at the end of the module.

    **Disclaimer: copied from
        firedrake/tests/regression/test_adjoint_operators.py
    """
    if not pyadjoint.annotate_tape():
        pyadjoint.continue_annotation()
    yield
    if pyadjoint.annotate_tape():
        pyadjoint.pause_annotation()


def empty_get_function_spaces(mesh):
    return {}


def empty_get_form(mesh_seq):
    def form(index, sols):
        return {}

    return form


def empty_get_bcs(mesh_seq):
    def bcs(index):
        return []

    return bcs


def empty_get_solver(mesh_seq):
    def solver(index, ic):
        return {}

    return solver
