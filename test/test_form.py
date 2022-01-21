from firedrake import *
from pyroteus.form import transfer_form
import numpy as np


def test_transfer_linear_system():
    """
    Given a linear problem defined on some mesh, verify that
    solving the system generated by transferring it onto a
    uniformly refined mesh coincides with the solution of the
    linear problem defined on that mesh.
    """
    base_mesh, refined_mesh = MeshHierarchy(UnitSquareMesh(10, 10), 1)
    sp = {'ksp_type': 'cg', 'pc_type': 'sor'}

    def setup_problem(mesh):
        V = FunctionSpace(mesh, 'CG', 1)
        x, y = SpatialCoordinate(mesh)
        nu = Constant(1.0)
        f = (1 + 8*pi**2)*sin(2*pi*x)*sin(2*pi*y)
        u = TrialFunction(V)
        v = TestFunction(V)
        yield nu*inner(grad(u), grad(v))*dx + u*v*dx
        yield f*v*dx
        yield Function(V)

    # Get the forms in the base space
    a, L, uh = setup_problem(base_mesh)

    # Transfer them to the refined space and solve
    a_plus = transfer_form(a, refined_mesh)
    L_plus = transfer_form(L, refined_mesh)
    V_plus = FunctionSpace(refined_mesh, 'CG', 1)
    uh_plus = Function(V_plus)
    solve(a_plus == L_plus, uh_plus, solver_parameters=sp)

    # Redefine in the refined space, solve and compare
    a, L, uh = setup_problem(refined_mesh)
    solve(a == L, uh, solver_parameters=sp)
    assert np.isclose(errornorm(uh_plus, uh), 0.0)
